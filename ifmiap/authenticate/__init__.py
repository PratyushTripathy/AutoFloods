 # ifmiap/authenticate/__init__.py

import os

import pystac_client
import planetary_computer

def sign_in():
    """
        Authenticates the user and returns a catalog object for accessing STAC data.

        Returns:
            catalog (pystac_client.Client): A catalog object providing access to STAC data.

    """
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    return catalog
