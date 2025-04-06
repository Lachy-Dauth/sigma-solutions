from datamodel import OrderDepth, Trade, TradingState, Order, ProsperityEncoder, Symbol
from typing import Dict, List, Tuple
import json

class AutomatedMarketMaker:
    def __init__(self):
        # Initialize state
        self.position = {"RAINFOREST_RESIN": 0, "KELP": 0}
        self.position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50}
        
        # Initial pool size and parameters
        self.pool = {"RAINFOREST_RESIN": 1000, "KELP": 1000}  # Initial liquidity pool
        self.k = self.pool["RAINFOREST_RESIN"] * self.pool["KELP"]  # Constant product formula
        
        # Trading parameters
        self.max_order_size = 10  # Maximum size per order
        self.price_adjustment = 0.01  # Small adjustment to ensure profitability
        
        # Track price history
        self.price_history = {"RAINFOREST_RESIN": [], "KELP": []}

    def compute_fair_price(self, product: str) -> float:
        """Calculate fair price based on the constant product formula"""
        if product == "RAINFOREST_RESIN":
            return self.pool["KELP"] / self.pool["RAINFOREST_RESIN"]
        else:  # KELP
            return self.pool["RAINFOREST_RESIN"] / self.pool["KELP"]

    def update_pool(self, product: str, amount: int):
        """Update the pool after a trade"""
        if product == "RAINFOREST_RESIN":
            self.pool["RAINFOREST_RESIN"] += amount
            # Recalculate the amount of the other product to maintain k
            self.pool["KELP"] = self.k / self.pool["RAINFOREST_RESIN"]
        else:  # KELP
            self.pool["KELP"] += amount
            # Recalculate the amount of the other product to maintain k
            self.pool["RAINFOREST_RESIN"] = self.k / self.pool["KELP"]

    def can_trade(self, product: str, amount: int) -> bool:
        """Check if trade is within position limits"""
        new_position = self.position[product] + amount
        return abs(new_position) <= self.position_limits[product]

    def determine_orders(self, product: str, order_depth: OrderDepth) -> List[Order]:
        """Determine orders based on AMM strategy"""
        orders = []
        fair_price = self.compute_fair_price(product)
        
        # Save price to history
        self.price_history[product].append(fair_price)
        
        # Calculate bid and ask prices with a small spread for profit
        bid_price = fair_price * (1 - self.price_adjustment)
        ask_price = fair_price * (1 + self.price_adjustment)
        
        # Buy orders (we place bids)
        for ask_price_level, ask_volume in order_depth.sell_orders.items():
            # If market price is below our calculated fair bid price, we buy
            if ask_price_level < bid_price:
                # Ensure we don't exceed position limit
                buy_volume = min(-ask_volume, self.position_limits[product] - self.position[product], self.max_order_size)
                if buy_volume > 0:
                    orders.append(Order(product, ask_price_level, buy_volume))
                    # Update our theoretical position and pool
                    self.position[product] += buy_volume
                    self.update_pool(product, buy_volume)
        
        # Sell orders (we place asks)
        for bid_price_level, bid_volume in order_depth.buy_orders.items():
            # If market price is above our calculated fair ask price, we sell
            if bid_price_level > ask_price:
                # Ensure we don't exceed position limit
                sell_volume = min(bid_volume, self.position[product] + self.position_limits[product], self.max_order_size)
                if sell_volume > 0:
                    orders.append(Order(product, bid_price_level, -sell_volume))
                    # Update our theoretical position and pool
                    self.position[product] -= sell_volume
                    self.update_pool(product, -sell_volume)
        
        return orders

# Main trader class that will be used by the exchange
class Trader:
    def __init__(self):
        self.amm = AutomatedMarketMaker()
        
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Main method called by the exchange to get the trader's orders.
        
        :param state: The current state of the market
        :return: A tuple containing:
            - A dictionary mapping product names to lists of Orders
            - A dictionary defining conversion orders
            - Trader data as a JSON string to be passed in next time
        """
        result = {}
        
        # Load previous state from trader_data if available
        if state.traderData:
            try:
                data = json.loads(state.traderData)
                if "pool" in data:
                    self.amm.pool = data["pool"]
                    self.amm.k = self.amm.pool["RAINFOREST_RESIN"] * self.amm.pool["KELP"]
                if "price_history" in data:
                    self.amm.price_history = data["price_history"]
            except:
                # If there's an error parsing the data, just use defaults
                pass
        
        # Process each product
        for product in ["RAINFOREST_RESIN", "KELP"]:
            # Skip if this product isn't in the current market data
            if product not in state.order_depths:
                continue
                
            # Get market data for this product
            order_depth = state.order_depths[product]
            
            # Update position with the latest from the exchange
            if product in state.position:
                self.amm.position[product] = state.position[product]
            
            # Compute orders for this product
            orders = self.amm.determine_orders(product, order_depth)
            
            # Add to result if we have orders
            if orders:
                result[product] = orders
        
        # Store current state info in trader_data as JSON string
        trader_data = {
            "pool": self.amm.pool,
            "position": self.amm.position,
            "price_history": self.amm.price_history
        }
        
        # Serialize to JSON string
        trader_data_json = json.dumps(trader_data)
        
        conversions = 1
        
        return result, conversions, trader_data_json
    
    def __str__(self) -> str:
        return f"AMM Trader with positions: {self.amm.position}"
    
    def __repr__(self) -> str:
        return self.__str__()
