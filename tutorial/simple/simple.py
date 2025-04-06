from datamodel import OrderDepth, Trade, TradingState, Order, ProsperityEncoder, Symbol
from typing import Dict, List, Tuple
import json

class Trader:
    def __init__(self):
        # The spread we want to maintain between our buy and sell prices
        self.spread = 2
        # Position limits
        self.position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50}
        # Order size to place each time
        self.order_size = 5
        
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], Dict[Symbol, int], str]:
        """
        Simple market making strategy:
        1. Find the mid price between best bid and best ask
        2. Place buy orders a bit below mid price
        3. Place sell orders a bit above mid price
        4. Lean towards selling when we have too much inventory (positive position)
        5. Lean towards buying when we have negative inventory (negative position)
        """
        result = {}
        
        # Process each product dynamically from state.order_depths
        for product in state.order_depths.keys():
            # Get market data
            order_depth = state.order_depths[product]
            
            # Get our current position (defaults to 0 if not in state)
            position = state.position.get(product, 0)
            
            # Set position limit (default to 50 if not explicitly defined)
            position_limit = self.position_limits.get(product, 50)
            
            # Only proceed if there are buy and sell orders in the market
            if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
                # Find best bid and ask prices
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                
                # Calculate mid price
                mid_price = (best_bid + best_ask) / 2
                
                # Calculate our buy and sell prices with the spread
                our_bid = int(mid_price - self.spread / 2)
                our_ask = int(mid_price + self.spread / 2) + 1
                
                orders = []
                
                # Position bias - adjust our order sizes based on current position
                buy_size = self.order_size
                sell_size = self.order_size
                
                # Calculate position ratio (-1.0 to 1.0)
                position_ratio = position / position_limit
                
                # Adjust buy/sell sizes based on position
                buy_size = int(self.order_size * (1 - position_ratio))
                sell_size = int(self.order_size * (1 + position_ratio))
                
                # Ensure minimum size is at least 1
                buy_size = max(1, buy_size)
                sell_size = max(1, sell_size)
                
                # Check if buying would exceed position limit
                if position + buy_size <= position_limit:
                    print(f"Placing buy order for {product}: Price {our_bid}, Size {buy_size}")
                    orders.append(Order(product, our_bid, buy_size))
                
                # Check if selling would exceed negative position limit
                if position - sell_size >= -position_limit:
                    print(f"Placing sell order for {product}: Price {our_ask}, Size {sell_size}")
                    orders.append(Order(product, our_ask, -sell_size))
                
                # Add orders to result
                result[product] = orders
        
        # We don't need complex trader data for this simple strategy
        trader_data = json.dumps({"timestamp": state.timestamp})
        
        conversions = 1
        
        return result, conversions, trader_data