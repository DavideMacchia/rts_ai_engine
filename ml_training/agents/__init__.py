"""
Agent tiers for the RTS AI hierarchy (see docs/architecture.md).

    common/    tier-agnostic building blocks (extractors, callbacks, BC + critic warm-up)
    district/  the settlement / district-tier agent  <- what exists today
    macro/     the empire-level brain (founds districts, allocates)  <- not built
    camp/      the military-camp tier  <- not built
"""
