import pandas as pd
import os
ns=[1192,1257,1604]


found_separator = False
stores=[]
os.chdir(r"H:\My Drive\Advanced_Simulation\HW5\Model2-Q3")
print(os.getcwd())

for n in ns:
    # master=pd.read_csv(r'H:\My Drive\Advanced_Simulation\HW5\Model2-Q3\HD_LOW.xlsx')
    data=pd.read_excel('HD_LOW.xlsx')
    # print(data)
    gurobi="gurobi_output"+str(n)+"_999999.log"
    anyloigc=pd.read_csv("Locations"+str(n)+".csv")


    with open(gurobi, 'r') as file:
        for line in file:
            # Check if the line contains "Best objective"
            if "Best objective" in line:
                print(line)
                best_objective=line.split(" ")[2].replace(",","")
            
            if line.strip() == '-------------------------':
                found_separator = True
                continue  # Skip the separator line
            if found_separator:
                parts = line.split()
                print(parts)
                
                if any("Y" in element for element in parts):
                    #NOW UNDER THE Y's
                    part=line.replace("[","").replace("]","").split("            ")
                    # print(part)
                    # input()
                    part=part[0].replace("Y","")
                    
                    stores.append(part)
            
        data_stores = {'store': stores}
        data_stores = pd.DataFrame(data_stores)

        # data_stores_lines['store']=data_stores_lines['store'].astype(int)-1

        data = data[(data['lat'] > 25) & (data['lat'] < 49) & (data['lon'] > -124.8) & (data['lon'] < -66.9)]
        data=data.drop_duplicates(subset='index',keep='first')
        data=data[['index','LocID', 'HD', 'LOW', 'city', 'STATE_NAME', 'population','lat', 'lon']]
        print(data.shape)
        # data['SUM'] = data['HD'] + data['LOW']


        data=data.merge(data_stores,left_on='index',right_on='store',how='left').reset_index()

        print(data)
