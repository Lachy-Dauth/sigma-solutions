from datamodel import OrderDepth, Trade, TradingState, Order, ProsperityEncoder, Symbol
from typing import Dict, List, Tuple
import json
import math

class Trader:
    def __init__(self):
        # Dictionary to store price history for each product
        self.price_history = {}
        # Window size for volatility calculation
        self.window_size = 10
        # Minimum spread to maintain
        self.min_spread = 1
        # Multiplier for volatility to determine spread
        self.volatility_multiplier = 2.0
        # Position limits
        self.position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50}
        # Base order size to place at the best prices
        self.base_order_size = 5
        # Number of order layers to place
        self.num_layers = 3
        # Size multiplier for each layer away from mid price
        self.size_multiplier = 1.5
        # Price increment between layers (as a multiple of volatility)
        self.layer_spacing_factor = 1.0
        # Store last mid prices to calculate volatility
        self.last_mid_prices = {}
        
    def calculate_volatility(self, product: str, current_mid: float) -> float:
        """
        Calculate the volatility of a product based on recent price history
        Returns the standard deviation of price changes
        """
        # Initialize price history for new products
        if product not in self.price_history:
            self.price_history[product] = []
            
        # Add current price to history
        self.price_history[product].append(current_mid)
        
        # Keep only the latest window_size prices
        if len(self.price_history[product]) > self.window_size:
            self.price_history[product].pop(0)
        
        # Need at least 2 data points to calculate volatility
        if len(self.price_history[product]) < 2:
            return 1.0  # Default volatility when we don't have enough data
        
        # Calculate price changes
        price_changes = [abs(self.price_history[product][i] - self.price_history[product][i-1]) 
                         for i in range(1, len(self.price_history[product]))]
        
        # Calculate mean of price changes
        mean_change = sum(price_changes) / len(price_changes)
        
        # Calculate standard deviation (volatility)
        if len(price_changes) > 1:
            variance = sum((x - mean_change) ** 2 for x in price_changes) / len(price_changes)
            volatility = math.sqrt(variance)
        else:
            volatility = mean_change  # If only one change, use that as volatility
            
        return max(1.0, volatility)  # Ensure minimum volatility of 1.0
        
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], Dict[Symbol, int], str]:
        """
        Layered market making strategy with dynamic spread based on volatility:
        1. Calculate volatility for each product
        2. Set spread proportional to the product's volatility
        3. Place multiple layers of orders with increasing sizes further from mid price
        4. Adjust order sizes based on current position
        """
        result = {}
        last_mid_prices = {}
        
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
                last_mid_prices[product] = mid_price
                
                # Calculate volatility and determine spread
                volatility = self.calculate_volatility(product, mid_price)
                dynamic_spread = max(self.min_spread, int(volatility * self.volatility_multiplier))
                
                # Calculate layer spacing based on volatility
                layer_spacing = max(1, int(volatility * self.layer_spacing_factor))
                
                orders = []
                
                # Position bias - adjust our base order sizes based on current position
                position_ratio = position / position_limit  # Range: -1.0 to 1.0
                
                # Calculate remaining capacity for buy/sell
                buy_capacity = position_limit - position
                sell_capacity = position_limit + position
                
                # Place layered orders
                for layer in range(self.num_layers):
                    # Calculate layer-specific spread (increases with each layer)
                    layer_spread = dynamic_spread + (layer * layer_spacing * 2)
                    
                    # Calculate prices for this layer
                    layer_bid = int(mid_price - layer_spread / 2) - (layer * layer_spacing)
                    layer_ask = int(mid_price + layer_spread / 2) + (layer * layer_spacing) + 1
                    
                    # Calculate base order size for this layer
                    # Orders get larger as they move away from mid price
                    layer_size = int(self.base_order_size * (self.size_multiplier ** layer))
                    
                    # Adjust sizes based on position
                    buy_size = int(layer_size * (1 - 0.5 * position_ratio))  # Buy less when position is high
                    sell_size = int(layer_size * (1 + 0.5 * position_ratio))  # Sell more when position is high
                    
                    # Ensure minimum size is at least 1
                    buy_size = max(1, buy_size)
                    sell_size = max(1, sell_size)
                    
                    # Make sure we don't exceed position limits
                    buy_size = min(buy_size, buy_capacity)
                    sell_size = min(sell_size, sell_capacity)
                    
                    # Only place orders if the size is valid
                    if buy_size > 0:
                        print(f"Placing buy order for {product}: Price {layer_bid}, Size {buy_size}, Layer {layer+1}")
                        orders.append(Order(product, layer_bid, buy_size))
                        buy_capacity -= buy_size
                    
                    if sell_size > 0:
                        print(f"Placing sell order for {product}: Price {layer_ask}, Size {sell_size}, Layer {layer+1}")
                        orders.append(Order(product, layer_ask, -sell_size))
                        sell_capacity -= sell_size
                    
                    # Stop placing orders if we've reached position limits
                    if buy_capacity <= 0 and sell_capacity <= 0:
                        break
                
                # Add orders to result
                result[product] = orders
        
        # Store current state information for future reference
        trader_data = json.dumps({
            "timestamp": state.timestamp,
            "price_history": self.price_history
        })
        
        conversions = 1
        
        return result, conversions, trader_data