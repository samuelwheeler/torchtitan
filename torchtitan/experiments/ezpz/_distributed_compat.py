import ezpz

try:
    import ezpz.distributed as ezpz_distributed
except ModuleNotFoundError:
    import ezpz.dist as ezpz_distributed
    ezpz.distributed = ezpz_distributed

__all__ = ["ezpz_distributed"]
