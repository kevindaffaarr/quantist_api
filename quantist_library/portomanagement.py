from decimal import Decimal

# CONSTANTS
LOT_SIZE = 100

class StockPorto():
    def __init__(self,
        current_porto: Decimal = Decimal(0),
        risk_aversion: float = 2.0,
        ):
        self.current_porto: Decimal = current_porto
        self.risk_aversion: float = risk_aversion

    def adj_porto(self,
        value: int,
        ) -> Decimal:
        """
        Function to adjust the current portfolio value
        If you want to add money, use positive value
        If you want to remove money, use negative value
        """
        self.current_porto += Decimal(value)
        return self.current_porto

    def position_sizing(self,
        price_current: int,
        price_target: int|None = None,
        price_cl: int|None = None,
        ) -> tuple[int, int]:
        """
        Money Management with Risk-Based Method.
        The risk_aversion number is adjusted by reward-to-risk ratio to output the lot and value sizing of the portofolio
        """
        # Calculate reward-to-risk ratio
        rtr:float
        if price_target is not None and price_cl is not None:
            rtr = (price_target - price_current) / (price_cl - price_current)
        else:
            rtr = 1
        
        # Calculate adjusted risk_aversion
        adj_risk_aversion:float = self.risk_aversion/100 * rtr

        # Calculate the risked portion of the portfolio
        lot:int = int(self.current_porto * Decimal(adj_risk_aversion) / price_current / LOT_SIZE)
        value:int = int(lot * price_current * LOT_SIZE)
        
        return lot, value