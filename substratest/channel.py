import dataclasses
import json
import time
import typing

import substra

from . import errors

# fields overriden by the backend that must be stripped in order to anonymize the asset
_OVERRIDDEN_FIELDS = (
    # storage address is replaced by each backend with its internal address
    "storage_address",
    "host",
)


def _anonymize_asset(asset: dict) -> dict:
    """Anonymize asset from an organization point of view."""

    anonymized_asset = dict(asset)  # make a copy so as not to modify input args
    for field, value in asset.items():
        if field in _OVERRIDDEN_FIELDS:
            anonymized_asset[field] = None
        elif isinstance(value, list):
            anonymized_asset[field] = [_anonymize_asset(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict):
            anonymized_asset[field] = _anonymize_asset(value)
    return anonymized_asset


@dataclasses.dataclass
class Channel:
    """Represents a channel, that is to say a group of users belonging to the same channel."""

    clients: typing.List[substra.Client]
    organization_sync_timeout: int

    def wait_for_asset_synchronized(self, asset):
        """Check if an asset is synchronized in all organizations.

        Asset are synchronized through orchestrator event in all organization, as a
        result asset creation and update is asynchronous accross organizations.

        This method is needed only in tests as in production user should only interact with
        its own organization.
        """
        unsynchronized_clients = {c.organization_id: c for c in self.clients}
        unsynchronized_assets = {c.organization_id: None for c in self.clients}

        reference_asset = _anonymize_asset(asset.dict())

        tstart = time.time()

        while True:
            if time.time() - tstart > self.organization_sync_timeout:
                # provide representation of unsynchronized assets as it will help
                # to understand quickly a test failure
                raise errors.SynchronizationTimeoutError(
                    f"Asset {asset.key} not synchronized: "
                    f"reference_asset={reference_asset}; "
                    f"unsynchronized_assets={json.dumps(unsynchronized_assets, default=str)}"
                )

            # iterate on a copy of the dict as its items are modified inside the for loop
            for organization_id, client in list(unsynchronized_clients.items()):
                try:
                    local_asset = client.get(asset)
                except substra.exceptions.NotFound:
                    continue

                anonymized_asset = _anonymize_asset(local_asset.dict())

                if anonymized_asset == reference_asset:
                    del unsynchronized_clients[organization_id]
                    del unsynchronized_assets[organization_id]
                    continue

                # store the latest state of unsynchronized asset for debugging purposes
                unsynchronized_assets[organization_id] = anonymized_asset

            # all backends have the same representation of the asset, synchronization is done
            if not unsynchronized_clients:
                return

            time.sleep(0.5)
