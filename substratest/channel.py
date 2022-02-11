import dataclasses
import time
import typing

import substra

from . import cfg
from . import errors


@dataclasses.dataclass
class Channel:
    """Represents a channel, that is to say a group of users belonging to the same channel."""

    clients: typing.List[substra.Client] = dataclasses.field(default_factory=list)

    def wait_for_asset_synchronized(self, asset, timeout=cfg.DELAY_ORGANISATIONS_SYNCHRONIZATION):
        """Check if an asset is synchronized in all organisations.

        Asset are synchronized through orchestrator event in all organisation, as a
        result asset creation and update is asynchronous accross organisations.

        This method is needed only in tests as in production user should only interact with
        its own organization.
        """
        unsynchronized_clients = {c.node_id: c for c in self.clients}

        tstart = time.time()

        while True:
            if time.time() - tstart > timeout:
                raise errors.SynchronizationTimeoutError(
                    f"Asset {asset} not synchronized on nodes {','.join(unsynchronized_clients.keys())}"
                )

            # iterate on a copy of the dict as we modify its items inside the for loop
            for node_id, client in list(unsynchronized_clients.items()):
                try:
                    local_asset = client.get(asset)
                except substra.exceptions.NotFound:
                    local_asset = None

                if local_asset:
                    # FIXME to handle asset update we will need to check that the asset
                    # content is the same from all organisations. This is not
                    # straightforward as backend of each node is redefining the
                    # address field for instance.
                    del unsynchronized_clients[node_id]

            if not unsynchronized_clients:
                return

            time.sleep(0.5)
