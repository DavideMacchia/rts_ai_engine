"""
Resource management for the RTS simulator.
Handles resource production and consumption.

Production and consumption are per-district: a district's buildings are staffed by
its own population and draw on its own stock. Transfers between districts are
Stage 4 (logistics).
"""

from ..game_state import District, Faction
from ..config import (
    PRODUCTION_RATES,
    RESOURCE_PROCESSING,
    POPULATION_FOOD_CONSUMPTION,
    POPULATION_WATER_CONSUMPTION,
    EDIBLE_FOODS,
)


class ResourceManager:
    """Manages resource production and consumption."""

    def produce_resources(self, faction: Faction, delta_time: float):
        """
        Produce resources for every district of the faction.

        Args:
            faction: The faction to produce resources for
            delta_time: Time elapsed in seconds
        """
        for district in faction.districts:
            self.produce_district_resources(district, delta_time)

    def produce_district_resources(self, district: District, delta_time: float):
        """
        Produce resources continuously, consuming inputs when required.

        Args:
            district: The district to produce resources for
            delta_time: Time elapsed in seconds
        """
        for building_type, count in district.buildings.items():
            # What a building yields depends ONLY on WHO staffs it: the number of workers and
            # their skill (D20). Where it stands does not change its output — the map limits a
            # district by how many buildings fit on its plot, not by taxing their yield.
            productivity = district.get_building_productivity(building_type)

            # Skip production if no workers
            if productivity <= 0.0:
                continue

            # Check if this building requires resource processing (inputs -> outputs)
            if building_type in RESOURCE_PROCESSING:
                # Processing building - must consume inputs to produce outputs
                processing = RESOURCE_PROCESSING[building_type]
                inputs_needed = processing.get('input', {})
                outputs = processing.get('output', {})

                # Process for each building of this type
                for _ in range(count):
                    # Calculate actual input needed for this time period
                    # Note: RESOURCE_PROCESSING rates are per turn (1 hour = 3600 seconds)
                    time_ratio = delta_time / 3600.0

                    # Check if we have all required inputs
                    has_inputs = True
                    for input_resource, input_amount in inputs_needed.items():
                        needed = input_amount * time_ratio
                        if district.get_resource(input_resource) < needed:
                            has_inputs = False
                            break

                    # If we have inputs, consume them and produce outputs
                    if has_inputs:
                        # Consume inputs
                        for input_resource, input_amount in inputs_needed.items():
                            needed = input_amount * time_ratio
                            district.remove_resource(input_resource, needed)

                        # Produce outputs (scaled by productivity)
                        for output_resource, output_amount in outputs.items():
                            produced = output_amount * time_ratio * productivity
                            district.add_resource(output_resource, produced)

            elif building_type in PRODUCTION_RATES:
                # Raw resource production - no inputs needed
                production = PRODUCTION_RATES[building_type]

                for resource, rate_per_hour in production.items():
                    rate_per_second = rate_per_hour / 3600.0
                    amount = rate_per_second * delta_time * count * productivity
                    district.add_resource(resource, amount)

    def consume_resources(self, faction: Faction, delta_time: float):
        """
        Consume resources for every district of the faction.

        Args:
            faction: The faction to consume resources for
            delta_time: Time elapsed in seconds
        """
        for district in faction.districts:
            self.consume_district_resources(district, delta_time)
            self.consume_clothes(district, delta_time)

    def consume_clothes(self, district: District, delta_time: float):
        """Clothes wear out. This is what makes the tailor's trade permanent rather than a
        one-off purchase: a district stays clothed only while somebody keeps making them."""
        from ..config import CLOTHES_CONSUMPTION_PER_CAPITA
        worn = CLOTHES_CONSUMPTION_PER_CAPITA * district.population * delta_time / 3600.0
        if worn > 0:
            district.remove_resource('clothes', min(worn, district.get_resource('clothes')))

    def consume_district_resources(self, district: District, delta_time: float):
        """
        Consume resources (food/water). A district feeds its own population.

        The `minimum_food` policy (lever L1, affordance) rations what each sim may
        draw. It constrains the RESOURCE, never the action: sims still choose to eat,
        they just find less of it, and pay for it in happiness. Banning a vital need
        would deadlock the population — see invariant I1.

        Args:
            district: The district to consume resources for
            delta_time: Time elapsed in seconds
        """
        if district.population <= 0:
            return

        # Food consumption, rationed by policy. The population eats from every edible
        # food in proportion to what is in stock, so keeping several foods around
        # spreads consumption and sustains the variety bonus. Grain and flour are not
        # edible (see EDIBLE_FOODS) — a district on grain alone starves.
        food_per_second = POPULATION_FOOD_CONSUMPTION / 3600.0
        food_needed = (
            food_per_second * district.population * delta_time
            * district.policies.ration_factor()
        )

        available = {f: district.get_resource(f) for f in EDIBLE_FOODS}
        total_available = sum(available.values())
        if total_available > 0:
            eaten = min(food_needed, total_available)
            for food, amount in available.items():
                district.remove_resource(food, eaten * amount / total_available)

        # Water consumption
        water_per_second = POPULATION_WATER_CONSUMPTION / 3600.0
        water_needed = water_per_second * district.population * delta_time
        district.remove_resource('water', water_needed)
