#ifmiap/authenticate/__init__.py

import os, pystac_client, planetary_computer

def sign_in():
    """
    Authenticates the user and returns a catalog object for accessing STAC data.

    Returns:
    catalog (pystac_client.Client): A catalog object providing access to STAC data.

    """
    planetary_computer.settings.set_subscription_key('ecc3eacf71734a14b9ea9053efb34ed5')

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    return catalog
