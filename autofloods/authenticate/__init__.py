#autofloods/authenticate/__init__.py

"""
sign_in() is deprecated: use autofloods.sources.MPCSource instead. It
is kept importable for backward compatibility but is no longer called
internally by autofloods.flood_mapper. setup_earthdata_login() is NOT
deprecated -- it is the recommended way to set up NASA Earthdata Login
credentials for autofloods.sources.OPERASource.
"""

import getpass
import os
import platform

import pystac_client, planetary_computer

_EARTHDATA_MACHINE = "urs.earthdata.nasa.gov"


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


def _read_netrc_text(path):
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return ""


def _remove_existing_machine_block(text, machine):
    """
    Remove any existing `machine <machine> ...` block from `text`,
    stopping at the next top-level `machine`/`default` token or end of
    file -- everything else in `text` (other machine blocks, blank
    lines, ordering) is left untouched. Only handles the standard
    multi-line `machine`/`login`/`password` block form (what
    Getting Started's manual instructions also use), not `macdef`
    blocks or the compressed single-line form; a .netrc using either
    of those should be edited by hand instead.

    Returns text with the block removed.
    """
    lines = text.splitlines(keepends=True)
    out = []
    i = 0
    while i < len(lines):
        tokens = lines[i].split()
        if tokens[:2] == ["machine", machine]:
            i += 1
            while i < len(lines):
                next_tokens = lines[i].split()
                if next_tokens[:1] in (["machine"], ["default"]):
                    break
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return "".join(out)


def setup_earthdata_login(username=None, password=None, netrc_path=None, verify=True):
    """
    Interactively set up NASA Earthdata Login credentials for
    OPERASource, by writing (or updating) a urs.earthdata.nasa.gov
    entry in ~/.netrc.

    If username/password aren't given, prompts for a username via
    input() and a password via getpass.getpass() -- the password is
    never echoed to the terminal, never printed, and never logged.
    Passing them directly (e.g. from an environment variable or a
    secrets manager) skips the interactive prompt entirely, for
    scripted/non-interactive setup.

    Any existing urs.earthdata.nasa.gov block in the target .netrc is
    replaced in place; every other entry in the file (other machines,
    blank lines, ordering) is left untouched. If no .netrc file exists
    yet, one is created.

    After writing, the file's permissions are set to owner-read/write
    only (chmod 0o600) -- most .netrc-reading tools (including the
    requests-based OAuth flow OPERASource relies on) expect this and
    may otherwise ignore or reject the file. Skipped on Windows, where
    POSIX permission bits don't apply the same way; Windows users
    should rely on NTFS file/user-account permissions instead.

    Parameters
    ----------
    username : str, optional
        NASA Earthdata Login username. Prompted via input() if not
        given.
    password : str, optional
        NASA Earthdata Login password. Prompted via getpass.getpass()
        (not echoed) if not given.
    netrc_path : str, optional
        Path to the .netrc file to write/update. Defaults to
        os.path.expanduser('~/.netrc').
    verify : bool, optional
        If True (default), re-reads the file after writing and
        confirms the urs.earthdata.nasa.gov entry is present with the
        expected login -- catches an obvious write failure (e.g. a
        read-only filesystem) without making a network request. Does
        NOT validate the credentials themselves against NASA's servers.

    Returns
    -------
    str
        The path to the .netrc file that was written/updated.

    Raises
    ------
    RuntimeError
        If verify=True and the written file doesn't contain the
        expected entry on read-back.
    """
    if netrc_path is None:
        netrc_path = os.path.expanduser("~/.netrc")

    if username is None:
        username = input("NASA Earthdata Login username: ")
    if password is None:
        password = getpass.getpass("NASA Earthdata Login password: ")

    remaining_text = _remove_existing_machine_block(
        _read_netrc_text(netrc_path), _EARTHDATA_MACHINE
    ).rstrip("\n")
    new_block = f"machine {_EARTHDATA_MACHINE}\n  login {username}\n  password {password}\n"
    new_text = f"{remaining_text}\n{new_block}" if remaining_text else new_block

    with open(netrc_path, "w") as f:
        f.write(new_text)

    if platform.system() != "Windows":
        os.chmod(netrc_path, 0o600)

    if verify:
        written = _read_netrc_text(netrc_path)
        if f"machine {_EARTHDATA_MACHINE}" not in written or f"login {username}" not in written:
            raise RuntimeError(
                f"Failed to verify the {_EARTHDATA_MACHINE} entry was "
                f"written to {netrc_path}."
            )

    return netrc_path
