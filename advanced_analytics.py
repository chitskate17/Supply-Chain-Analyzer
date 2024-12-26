import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class AdvancedAnalytics:
    def __init__(self, data):
        self.data = data

    def arima_forecast(self):
        """Perform ARIMA forecasting on revenue over time without a date column."""
        if self.data is None or 'Revenue generated' not in self.data.columns:
            raise ValueError("Data must be provided for ARIMA forecasting and must contain 'Revenue generated'.")

        # Create a synthetic time index
        self.data['Time'] = range(len(self.data))  # Creating a sequential integer index
        df_revenue = self.data[['Time', 'Revenue generated']].set_index('Time')

        # Fit ARIMA model (p,d,q) parameters can be tuned based on your dataset characteristics
        model = sm.tsa.ARIMA(df_revenue['Revenue generated'], order=(1, 1, 1))
        results = model.fit()

        # Forecast future values
        forecast_steps = 12  # Forecasting next 12 periods (e.g., months)
        forecast = results.forecast(steps=forecast_steps)

        # Plotting the forecasted values
        plt.figure(figsize=(10, 5))
        plt.plot(df_revenue.index, df_revenue['Revenue generated'], label='Historical Revenue')

        # Create a new index for the forecasted values
        forecast_index = range(len(df_revenue), len(df_revenue) + forecast_steps)

        plt.plot(forecast_index, forecast, label='Forecasted Revenue', color='red')
        plt.title('Revenue Forecast using ARIMA')
        plt.xlabel('Time Index')
        plt.ylabel('Revenue Generated')
        plt.legend()
        plt.grid()

        # Save plot as an image file
        plt.savefig('static/images/revenue_forecast.png')
        plt.close()  # Close the plot to free memory

    def exponential_smoothing_forecast(self):
        """Perform Exponential Smoothing forecasting on revenue."""
        if self.data is None or 'Revenue generated' not in self.data.columns:
            raise ValueError(
                "Data must be provided for Exponential Smoothing forecasting and must contain 'Revenue generated'.")

        df_revenue = self.data[['Revenue generated']]

        # Fit Exponential Smoothing model
        model = ExponentialSmoothing(df_revenue['Revenue generated'], trend='add', seasonal=None)
        results = model.fit()

        # Forecast future values
        forecast_steps = 12  # Forecasting next 12 periods (e.g., months)
        forecast = results.forecast(steps=forecast_steps)

        # Plotting the forecasted values
        plt.figure(figsize=(10, 5))
        plt.plot(df_revenue.index, df_revenue['Revenue generated'], label='Historical Revenue')

        # Create a new index for the forecasted values
        forecast_index = range(len(df_revenue), len(df_revenue) + forecast_steps)

        plt.plot(forecast_index, forecast, label='Forecasted Revenue (Exponential Smoothing)', color='green')
        plt.title('Revenue Forecast using Exponential Smoothing')
        plt.xlabel('Time Index')
        plt.ylabel('Revenue Generated')
        plt.legend()
        plt.grid()

        # Save plot as an image file
        plt.savefig('static/images/exponential_smoothing_forecast.png')
        plt.close()  # Close the plot to free memory

    def analyze_turnover(self):
        """Analyze turnover by product type."""

        turnover_analysis = self.data.groupby('Product type').agg(
            total_revenue=('Revenue generated', 'sum'),
            total_sales=('Number of products sold', 'sum')
        ).reset_index()

        return turnover_analysis


    def analyze_profit_loss(self):
        """Identify profit-making and loss-making products along with top and bottom performers."""

        profit_loss_analysis = self.data.copy()

        # Check if 'Costs' column exists; if not, set a default value (e.g., 0)
        if 'Costs' not in profit_loss_analysis.columns:
            profit_loss_analysis['Costs'] = 0  # Default cost if not provided

        # Calculate Profit using existing columns: Profit = Revenue - Costs
        profit_loss_analysis['Profit'] = profit_loss_analysis['Revenue generated'] - profit_loss_analysis['Costs']

        profit_making_products = profit_loss_analysis[profit_loss_analysis['Profit'] > 0]
        loss_making_products = profit_loss_analysis[profit_loss_analysis['Profit'] <= 0]

        # Get top 10 and bottom 10 products based on revenue
        top_products = profit_loss_analysis.nlargest(10, 'Revenue generated')[['Product type', 'SKU', 'Revenue generated']]
        bottom_products = profit_loss_analysis.nsmallest(10, 'Revenue generated')[
            ['Product type', 'SKU', 'Revenue generated']]

        return (
            profit_making_products[['Product type', 'Profit']],
            loss_making_products[['Product type', 'Profit']],
            top_products,
            bottom_products
        )


    def identify_inefficiencies(self):
        """Identify inefficiencies in the supply chain."""

        inefficiency_analysis = self.data.copy()

        # Calculate Sales Availability Ratio and check for inefficiencies
        inefficiency_analysis['Sales Availability Ratio'] = inefficiency_analysis['Number of products sold'] / \
                                                            inefficiency_analysis['Availability']

        inefficient_products = inefficiency_analysis[
            inefficiency_analysis['Sales Availability Ratio'] < 0.5]  # Threshold can be adjusted

        return inefficient_products[['Product type', 'Sales Availability Ratio']]
