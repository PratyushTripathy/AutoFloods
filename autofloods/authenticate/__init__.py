#autofloods/authenticate/__init__.py

"""
Deprecated: use autofloods.sources.MPCSource instead. This module is kept
importable for backward compatibility but is no longer called internally
by autofloods.flood_mapper.
"""

import os

import pystac_client, planetary_computer

def sign_in():
    """
    Reads an optional MPC subscription key from the MPC_SUBSCRIPTION_KEY
    environment variable. A subscription key is never required -- MPC's
    STAC search and asset signing both work anonymously; a key only
    raises the request rate limit. The key is not validated here: an
    invalid or expired key is accepted silently and will only surface as
    an authentication error on a later search or asset-signing call.

    Returns:
        pystac_client.Client: client already configured to sign asset URLs.
    """
    subscription_key = os.environ.get('MPC_SUBSCRIPTION_KEY')
    if subscription_key:
        planetary_computer.settings.set_subscription_key(subscription_key)

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    return catalog
