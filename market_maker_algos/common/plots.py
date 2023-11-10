import matplotlib.pyplot as plt

def plot_result(history_info, plot_high_low=False):
    history_info.sort_values(by='datetime', inplace=True)
    datetime = history_info['datetime']

    f = plt.figure(figsize=(8, 15))
    f.add_subplot(3, 1, 1)
    plt.plot(datetime, history_info['close'], color='black', label='Mid-market price')
    plt.plot(datetime, history_info['reserve_price'], color='blue', linestyle='dashed', label='Reservation price')
    plt.plot(datetime, history_info['ask_price'], color='red', linestyle='', marker='.', label='Price asked', markersize=2)
    plt.plot(datetime, history_info['bid_price'], color='green', linestyle='', marker='.', label='Price bid', markersize=2)
    
    if set(['high', 'low']).issubset(set(history_info.columns)) and plot_high_low:
        plt.plot(datetime, history_info['high'], color='black', linestyle='dotted', label='High')
        plt.plot(datetime, history_info['low'], color='black', linestyle='dotted', label='Low')

    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Price', fontsize=16)
    plt.grid(True)
    plt.plot(
        'datetime',
        'bid_price',
        data=history_info[history_info.matched_bid_quantity > 0],
        color='green',
        linestyle='',
        marker='x',
        markersize=4,
        label='Bid matched'
    )
    plt.plot(
        'datetime',
        'ask_price',
        data=history_info[history_info.matched_ask_quantity > 0],
        color='red',
        linestyle='',
        marker='x',
        markersize=4,
        label='Ask matched'
    )
    plt.legend()

    f.add_subplot(3, 1, 2)
    plt.plot(datetime, history_info['nav'], color='black', label='NAV')
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('PnL', fontsize=16)
    plt.grid(True)
    plt.legend()

    f.add_subplot(3, 1, 3)
    plt.plot(datetime, history_info['quantity'], color='black', label='Stocks held')
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Inventory', fontsize=16)
    plt.grid(True)
    plt.legend()

    plt.show()