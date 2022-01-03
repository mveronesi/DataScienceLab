# DATASET LOADING
import csv
import datetime as dt
import matplotlib.pyplot as plt

dataset = []
date_format = "%Y-%m-%d"
with open("../datasets/GLT_filtered.csv") as f:
    next(f) # skip first row
    #measures = list(csv.reader(f))
    for row in csv.reader(f):
        dataset.append(
            {
                "date": dt.datetime.strptime(row[0], date_format),
                "averageTemperature": float(row[1]) if row[1] != '' else None,
                "averageTemperatureUncertainty": float(row[2]) if row[2] != '' else None,
                "city": row[3],
                "country": row[4],
                "latitude": row[5],
                "longitude": row[6]
            }
        )

dataset.sort(key = lambda item: (item["country"], item["city"], item["date"]))

# POINT 1: analyze the attribute "averageTemperature" for searching missing values
for i in range(len(dataset)):
    if dataset[i]["averageTemperature"] == None:      
        # looking for the previous value
        j = i-1
        prev = None
        while(j>=0 and dataset[i]["country"] == dataset[j]["country"] and dataset[i]["city"] == dataset[i]["city"] and prev == None):
            prev = dataset[j]["averageTemperature"] if dataset[j]["averageTemperature"] != None else None
            j -= 1

        if prev == None: # not found
            prev = 0
        
        # looking for the next value
        j = i+1
        next = None
        while(j<len(dataset) and dataset[i]["country"] == dataset[j]["country"] and dataset[i]["city"] == dataset[j]["city"] and next == None):
            next = dataset[j]["averageTemperature"] if dataset[j]["averageTemperature"] != None else None
            j += 1
        
        if next == None: # not found
            next = 0
            
        dataset[i]["averageTemperature"] = (prev + next) / 2

# POINT 2
def get_city(city_name, copy=True):
    first = None
    last = None
    for i in range(len(dataset)):
        if dataset[i]["city"] == city_name:
            first = i
            for j in range(i+1, len(dataset)):
                if(dataset[j]["city"] != dataset[i]["city"]):
                    last = j
                    break
            break
    return dataset[first:last].copy() if copy == True else dataset[first:last]

# POINT 3
def n_hottest(city_name, N):
    dataset_city = get_city(city_name)
    dataset_city.sort(reverse = True, key = lambda item: item["averageTemperature"])
    return dataset_city if len(dataset_city) <= N else dataset_city[0:N]

def n_coldest(city_name, N):
    dataset_city = get_city(city_name)
    dataset_city.sort(reverse = False, key = lambda item: item["averageTemperature"])
    return dataset_city if len(dataset_city) <= N else dataset_city[0:N]

# POINT 4
rome = get_city("Rome", copy=False)
bangkok = get_city("Bangkok", copy=False)

rome_temps = []
bangkok_temps = []
dates = []
for i in range(len(rome)):
    rome_temps.append(rome[i]["averageTemperature"])
    dates.append(rome[i]["date"])
    bangkok_temps.append(bangkok[i]["averageTemperature"])

plt.hist(rome_temps)
plt.hist(bangkok_temps)
plt.show()