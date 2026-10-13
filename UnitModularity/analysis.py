from scipy.stats import linregress
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt


def slope_each_station(station_df):  #this formula is to calculate a single slope for just one
    """Input: DataFrame with 'year' and 'data' for ONE station.
       Output: float, fitted slope (data units per year)."""
    result=linregress(station_df['year'],station_df['data'])
    return result.slope


def compute_all_slopes(df, min_n=40): #this formula is to calculate all slopes 
    """Input: cleaned table with many stations.
       Output: table with one row per station (state, name, lon, lat, slope, n)."""
    records=[] #to keep results

    for (state,name), station_df in df.groupby(['state','name']): 
        n=len(station_df)
        if n<min_n:
            continue
        slope=slope_each_station(station_df)

        records.append({
            'state': state,
            'name': name,
            'lon': station_df['lon'].iloc[0],
            'lat': station_df['lat'].iloc[0],
            'slope': slope,
            'n': n,
        })

    return pd.DataFrame(records)


def plot_histogram(slopes_df, label): # this formula is to plot histograms of the slopes
    """Input: table with a 'slope' column, and a label (text) for the title.
       Output: draws a histogram."""
    plt.hist(slopes_df['slope'], bins=30)
    plt.axvline(0, color='red')
    plt.title("Slopes of "+label+' vs year')
    plt.xlabel('Slope (change per year)')
    plt.ylabel("# of stations")
    plt.show()


def plot_map(slopes_df, label): #this is to plot maps 
    """Input: table with 'lon', 'lat', 'slope' columns, and a label (text).
       Output: saves a map as a .png file."""
    limit=slopes_df['slope'].abs().max()
    plt.scatter(slopes_df['lon'], slopes_df['lat'],
                c=slopes_df['slope'], cmap='coolwarm',
                vmin=-limit,vmax=limit, s=10)
    plt.colorbar(label='Slope (change per year)')
    plt.title('Slope of '+ label + ' vs year')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()