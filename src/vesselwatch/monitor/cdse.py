"""Finding and downloading Sentinel-2 imagery from the Copernicus Data Space Ecosystem.

Searching the catalogue needs no account. Downloading does: the login is read
from the CDSE_USERNAME and CDSE_PASSWORD environment variables and is never
written anywhere.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1"
DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
USER_ENV = "CDSE_USERNAME"
PASSWORD_ENV = "CDSE_PASSWORD"
TIMEOUT = 120  # seconds


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        return json.load(response)


def l1c_filter(tile: str, date: str) -> str:
    """OData filter for the Level-1C products of one MGRS tile sensed on one day (`YYYYMMDD`)."""
    day = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    return (f"Collection/Name eq 'SENTINEL-2' and contains(Name,'MSIL1C') and contains(Name,'_T{tile}_') "
            f"and ContentDate/Start ge {day}T00:00:00.000Z and ContentDate/Start le {day}T23:59:59.999Z")


def find_l1c_product(tile: str, date: str) -> dict:
    """The catalogue entry (`Id`, `Name`, ...) of the Level-1C product for a tile and day."""
    query = urllib.parse.urlencode({"$filter": l1c_filter(tile, date), "$top": 10})
    products = _get_json(f"{CATALOGUE}/Products?{query}")["value"]
    if not products:
        raise LookupError(f"No Sentinel-2 L1C product for tile {tile} on {date}")
    # Reprocessed copies of one acquisition differ only in the trailing processing time; take the newest
    return sorted(products, key=lambda p: p["Name"])[-1]


def _node_path(product: dict, parts) -> str:
    nodes = "".join(f"/Nodes({part})" for part in (product["Name"], *parts))
    return f"Products({product['Id']}){nodes}"


def list_nodes(product: dict, *parts) -> list[str]:
    """Names of the files and folders inside a product, below the given path."""
    return [node["Name"] for node in _get_json(f"{CATALOGUE}/{_node_path(product, parts)}/Nodes")["result"]]


def access_token(username: str | None = None, password: str | None = None) -> str:
    """A short-lived download token for the given or the configured login."""
    username = username or os.environ.get(USER_ENV)
    password = password or os.environ.get(PASSWORD_ENV)
    if not username or not password:
        raise ValueError(f"Set the {USER_ENV} and {PASSWORD_ENV} environment variables to download from Copernicus")
    form = urllib.parse.urlencode({"grant_type": "password", "client_id": "cdse-public",
                                   "username": username, "password": password}).encode()
    with urllib.request.urlopen(urllib.request.Request(TOKEN_URL, data=form), timeout=TIMEOUT) as response:
        return json.load(response)["access_token"]


def download_node(product: dict, parts, dest: Path, token: str) -> Path:
    """Download one file of a product. Written to a temporary name first, so a broken download leaves nothing behind."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(f"{DOWNLOAD}/{_node_path(product, parts)}/$value",
                                     headers={"Authorization": f"Bearer {token}"})
    partial = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response, open(partial, "wb") as f:
        shutil.copyfileobj(response, f, length=1 << 20)
    os.replace(partial, dest)
    return dest


def upper_left(tile_metadata: str) -> tuple[float, float]:
    """Map coordinates of the upper-left corner of the 10 m grid, from a product's MTD_TL.xml."""
    match = re.search(r'<Geoposition resolution="10">\s*<ULX>([-\d.]+)</ULX>\s*<ULY>([-\d.]+)</ULY>', tile_metadata)
    if not match:
        raise ValueError("No 10 m Geoposition in the tile metadata")
    return float(match.group(1)), float(match.group(2))


def fetch_true_colour(tile: str, date: str, dest_dir) -> tuple[Path, dict]:
    """Download the true-colour image (TCI, 10 m, 8-bit RGB) of one tile and day.

    Returns the image path and its georeference: product name and the map
    coordinates of the upper-left pixel corner. Files already in `dest_dir`
    are reused, so this needs the login only the first time.
    """
    dest_dir = Path(dest_dir)
    image = dest_dir / f"T{tile}_{date}_TCI.jp2"
    info_file = dest_dir / f"T{tile}_{date}.json"
    if image.exists() and info_file.exists():
        return image, json.loads(info_file.read_text())

    product = find_l1c_product(tile, date)
    granule = list_nodes(product, "GRANULE")[0]
    tci = next(name for name in list_nodes(product, "GRANULE", granule, "IMG_DATA") if name.endswith("_TCI.jp2"))

    metadata = download_node(product, ("GRANULE", granule, "MTD_TL.xml"), dest_dir / f"T{tile}_{date}_MTD_TL.xml",
                             access_token())
    ulx, uly = upper_left(metadata.read_text())
    # A fresh token per file: they expire after ten minutes and the image takes a while
    download_node(product, ("GRANULE", granule, "IMG_DATA", tci), image, access_token())

    info = {"product": product["Name"], "tile": tile, "date": date, "ulx": ulx, "uly": uly, "resolution": 10}
    info_file.write_text(json.dumps(info, indent=2))
    return image, info
