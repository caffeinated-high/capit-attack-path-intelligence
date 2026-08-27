"""
Central, dataset-agnostic configuration.

Every other module reads "which hosts matter" (high-value targets,
known-vulnerable hosts, subnet grouping) from here instead of hardcoding
it. This is what lets the exact same pipeline code run against either
the synthetic demo data or a real dataset (e.g. LANL) -- only the
values below change, nothing in capg/, cas/, models/, reasoning/,
optimization/ needs to be touched.

For a real deployment these values would come from your CMDB / asset
inventory / vulnerability scanner -- NOT be inferred from the traffic
logs themselves (that would be circular: you can't use "was this host
attacked" to decide "is this host valuable", you need that from an
independent source).
"""

# Hosts/computers considered high-value targets (domain controllers,
# databases holding sensitive data, etc.). Used by CAS objective
# inference, path reasoning, and the mission-level optimizer tier.
HIGH_VALUE_HOSTS: set = set()

# {host_id: cve_id} for hosts with a known vulnerability, e.g. from a
# vulnerability scanner export. Used to seed "exploit" edges in the CAPG.
VULNERABLE_HOSTS: dict = {}

# {host_id: subnet_label} used by the hierarchical optimizer's
# subnet-level intervention tier. If a host isn't listed, it falls back
# to "unknown".
HOST_SUBNET_MAP: dict = {}


def subnet_of(host: str) -> str:
    return HOST_SUBNET_MAP.get(host, "unknown")


def reset():
    """Used by the synthetic demo / tests to restore a clean config
    between runs, since the values above are mutated at load time."""
    global HIGH_VALUE_HOSTS, VULNERABLE_HOSTS, HOST_SUBNET_MAP
    HIGH_VALUE_HOSTS = set()
    VULNERABLE_HOSTS = {}
    HOST_SUBNET_MAP = {}
