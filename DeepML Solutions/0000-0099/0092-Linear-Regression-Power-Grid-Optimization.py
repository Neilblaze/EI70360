import math

PI = 3.14159


def power_grid_forecast(consumption_data):
    days = list(range(1, 11))
    n = len(days)

    detrended = [
        cons - 10 * math.sin((2 * PI * i) / 10)
        for i, cons in zip(days, consumption_data)
    ]

    sum_x = sum(days)
    sum_y = sum(detrended)
    sum_xy = sum(x * y for x, y in zip(days, detrended))
    sum_x2 = sum(x**2 for x in days)
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    b = (sum_y - m * sum_x) / n

    day_15_predictions = m * 15 + b + 10 * math.sin((2 * PI * 15) / 10)

    return math.ceil(round(day_15_predictions) * 1.05)


consumption_data = [150, 165, 185, 195, 210, 225, 240, 260, 275, 290]
y_pred = power_grid_forecast(consumption_data)
print(y_pred)
