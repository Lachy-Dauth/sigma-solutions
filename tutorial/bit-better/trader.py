from datamodel import OrderDepth, Trade, TradingState, Order, ProsperityEncoder, Symbol
from typing import Dict, List, Tuple
import json
import math

class ChangingPriceMM:
    """
    Class that implements a dynamic market making strategy with layered orders
    and volatility-based spread adjustment.
    """
    def __init__(self, product: str, position_limit: int = 50):
        self.product = product
        self.position_limit = position_limit
        self.price_history = []
        self.window_size = 10
        self.min_spread = 1
        self.volatility_multiplier = 2.0
        self.base_order_size = 5
        self.num_layers = 3
        self.size_multiplier = 1.5
        self.layer_spacing_factor = 1.0
        
    def calculate_volatility(self, current_mid: float) -> float:
        """
        Calculate the volatility based on recent price history
        Returns the standard deviation of price changes
        """
        # Add current price to history
        self.price_history.append(current_mid)
        
        # Keep only the latest window_size prices
        if len(self.price_history) > self.window_size:
            self.price_history.pop(0)
        
        # Need at least 2 data points to calculate volatility
        if len(self.price_history) < 2:
            return 1.0  # Default volatility when we don't have enough data
        
        # Calculate price changes
        price_changes = [abs(self.price_history[i] - self.price_history[i-1]) 
                         for i in range(1, len(self.price_history))]
        
        # Calculate mean of price changes
        mean_change = sum(price_changes) / len(price_changes)
        
        # Calculate standard deviation (volatility)
        if len(price_changes) > 1:
            variance = sum((x - mean_change) ** 2 for x in price_changes) / len(price_changes)
            volatility = math.sqrt(variance)
        else:
            volatility = mean_change  # If only one change, use that as volatility
            
        return max(1.0, volatility)  # Ensure minimum volatility of 1.0
    
    def generate_orders(self, order_depth: OrderDepth, current_position: int) -> List[Order]:
        """
        Generate layered orders based on market depth, volatility, and position
        """
        orders = []
        
        # Only proceed if there are buy and sell orders in the market
        if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
            return orders
            
        # Find best bid and ask prices
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        
        # Calculate mid price
        mid_price = (best_bid + best_ask) / 2
        
        # Calculate volatility and determine spread
        volatility = self.calculate_volatility(mid_price)
        dynamic_spread = max(self.min_spread, int(volatility * self.volatility_multiplier))
        
        # Calculate layer spacing based on volatility
        layer_spacing = max(1, int(volatility * self.layer_spacing_factor))
        
        # Position bias - adjust our base order sizes based on current position
        position_ratio = current_position / self.position_limit  # Range: -1.0 to 1.0
        
        # Calculate remaining capacity for buy/sell
        buy_capacity = self.position_limit - current_position
        sell_capacity = self.position_limit + current_position
        
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
                print(f"Placing buy order for {self.product}: Price {layer_bid}, Size {buy_size}, Layer {layer+1}")
                orders.append(Order(self.product, layer_bid, buy_size))
                buy_capacity -= buy_size
            
            if sell_size > 0:
                print(f"Placing sell order for {self.product}: Price {layer_ask}, Size {sell_size}, Layer {layer+1}")
                orders.append(Order(self.product, layer_ask, -sell_size))
                sell_capacity -= sell_size
            
            # Stop placing orders if we've reached position limits
            if buy_capacity <= 0 and sell_capacity <= 0:
                break
                
        return orders


class StablePriceMM:
    """
    Simple market maker for stable price assets that:
    1. Takes immediately profitable trades
    2. Undercuts existing orders by 1
    """
    def __init__(self, product: str, target_price: int, position_limit: int = 50):
        self.product = product
        self.target_price = target_price
        self.position_limit = position_limit
        # Amount to undercut by
        self.undercut_amount = 1
    
    def generate_orders(self, order_depth: OrderDepth, current_position: int) -> List[Order]:
        """
        Super simple strategy:
        1. If there are orders that are profitable relative to target price, take them
        2. Otherwise, just undercut the current best orders by 1
        """
        orders = []
        
        # Calculate remaining capacity
        buy_capacity = self.position_limit - current_position
        sell_capacity = self.position_limit + current_position
        
        # If there are no orders in the market, place orders at target price
        if not order_depth.buy_orders and not order_depth.sell_orders:
            if buy_capacity > 0:
                orders.append(Order(self.product, self.target_price - 1, buy_capacity))
            if sell_capacity > 0:
                orders.append(Order(self.product, self.target_price + 1, -sell_capacity))
            return orders
        
        # Check for immediately profitable trades first
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            # If someone is buying above our target price, we should sell to them
            if best_bid > self.target_price and sell_capacity > 0:
                # Take as much as we can of this profitable order
                sell_volume = min(abs(order_depth.buy_orders[best_bid]), sell_capacity)
                orders.append(Order(self.product, best_bid, -sell_volume))
                print(f"PROFITABLE SELL for {self.product}: Price {best_bid} > Target {self.target_price}, Size {sell_volume}")
                # Update capacity
                sell_capacity -= sell_volume
        
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            # If someone is selling below our target price, we should buy from them
            if best_ask < self.target_price and buy_capacity > 0:
                # Take as much as we can of this profitable order
                buy_volume = min(abs(order_depth.sell_orders[best_ask]), buy_capacity)
                orders.append(Order(self.product, best_ask, buy_volume))
                print(f"PROFITABLE BUY for {self.product}: Price {best_ask} < Target {self.target_price}, Size {buy_volume}")
                # Update capacity
                buy_capacity -= buy_volume
        
        # Simple undercutting strategy for the rest of our capacity
        if order_depth.buy_orders and buy_capacity > 0:
            best_bid = max(order_depth.buy_orders.keys())
            # Don't undercut if best bid is already at or above target price
            if best_bid < self.target_price - 1:
                # Undercut by placing a higher bid
                our_bid = best_bid + self.undercut_amount
                # But don't exceed target price
                our_bid = min(our_bid, self.target_price - 1)
                orders.append(Order(self.product, our_bid, buy_capacity))
                print(f"UNDERCUT BUY for {self.product}: Price {our_bid}, Size {buy_capacity}")
        
        if order_depth.sell_orders and sell_capacity > 0:
            best_ask = min(order_depth.sell_orders.keys())
            # Don't undercut if best ask is already at or below target price
            if best_ask > self.target_price + 1:
                # Undercut by placing a lower ask
                our_ask = best_ask - self.undercut_amount
                # But don't go below target price
                our_ask = max(our_ask, self.target_price + 1)
                orders.append(Order(self.product, our_ask, -sell_capacity))
                print(f"UNDERCUT SELL for {self.product}: Price {our_ask}, Size {sell_capacity}")
        
        return orders


class Trader:
    def __init__(self):
        # Dictionary to store market makers for each product
        self.market_makers = {}
        # Position limits for different products
        self.position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50}
        # Products that should use the StablePriceMM strategy with their target prices
        self.stable_products = {
            # Example: "PRODUCT_NAME": target_price
            "RAINFOREST_RESIN": 10000,
        }
        
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], Dict[Symbol, int], str]:
        """
        Main trading logic:
        1. Initialize or get market maker for each product
        2. Generate orders based on current market conditions
        """
        result = {}
        price_history_data = {}
        
        # Process each product dynamically from state.order_depths
        for product in state.order_depths.keys():
            # Get market data
            order_depth = state.order_depths[product]
            
            # Get our current position (defaults to 0 if not in state)
            position = state.position.get(product, 0)
            
            # Set position limit (default to 50 if not explicitly defined)
            position_limit = self.position_limits.get(product, 50)
            
            # Initialize market maker if it doesn't exist yet
            if product not in self.market_makers:
                # Use StablePriceMM for stable products, ChangingPriceMM for others
                if product in self.stable_products:
                    target_price = self.stable_products[product]
                    self.market_makers[product] = StablePriceMM(product, target_price, position_limit)
                else:
                    self.market_makers[product] = ChangingPriceMM(product, position_limit)
            
            # Generate orders using the appropriate market maker
            orders = self.market_makers[product].generate_orders(order_depth, position)
            
            # Add orders to result
            result[product] = orders
            
            # Store price history for logging if available
            if hasattr(self.market_makers[product], 'price_history'):
                price_history_data[product] = self.market_makers[product].price_history.copy()
        
        # Store current state information for future reference
        trader_data = json.dumps({
            "timestamp": state.timestamp,
            "price_history": price_history_data
        })
        
        conversions = 1
        
        return result, conversions, trader_data