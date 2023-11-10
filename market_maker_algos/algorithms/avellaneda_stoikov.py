import math

from .base_algorithm import Policy


class AvellanedaStoikov(Policy):
    def __init__(self, order_quantity):
        self.order_quantity = order_quantity

    def get_action(self, observation):
        (
            current_price,
            quantity,
            current_step,
            risk_factor,
            k,
            asset_sigma,
            total_time,
            dt,
        ) = observation

        # reserve price
        reserve_price = current_price - quantity * risk_factor * (asset_sigma**2) * (
            total_time - dt * current_step
        )
        # reserve spread
        reserve_spread = (
            risk_factor * asset_sigma**2 * (total_time - dt * current_step)
            + 2 / risk_factor * math.log(1 + risk_factor / k)
        )
        # print(reserve_spread)

        # optimal quotes
        bid_price = reserve_price - reserve_spread / 2
        ask_price = reserve_price + reserve_spread / 2

        action = (
            self.order_quantity,
            bid_price,
            self.order_quantity,
            ask_price,
        )
        return action, {"reserve_price": reserve_price}
