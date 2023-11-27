import pandas as pd




def format_csv(dataframe_path):

    initial_indices = []
    final_indices = []
    found_separator = False
    stores=[]

    # data=pd.read_csv(dataframe_path)
    with open(dataframe_path, 'r') as file:
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
                
                if any("X" in element for element in parts):
                    parts=parts[0].replace("X","").replace("[","").replace("]","").split(",")

                    initial_index = int(parts[0])
                    final_index = int(parts[1])
                    initial_indices.append(initial_index)
                    final_indices.append(final_index)
                
                elif any("Y" in element for element in parts):
                    #NOW UNDER THE Y's
                    part=line.replace("[","").replace("]","").split("            ")
                    # print(part)
                    # input()
                    part=part[0].replace("Y","")
                    
                    stores.append(part)



        # Create a DataFrame
        data = {'ini_store': initial_indices, 'end_store': final_indices}
        df_lines = pd.DataFrame(data)

        data_stores = {'store': stores}
        data_stores_lines = pd.DataFrame(data_stores)



        #RE ACCONDITION
        
        df_lines['ini_store']=df_lines['ini_store'].astype(int)-1
        df_lines['end_store']=df_lines['end_store'].astype(int)-1


        data_stores_lines['store']=data_stores_lines['store'].astype(int)-1


        #MERGE!


        #FIRST IMPORT:
        data=pd.read_excel(r'H:\My Drive\Advanced_Simulation\HW5\Model2-Q3\HD_LOW.xlsx')[['LocID', 'HD', 'LOW', 'city', 'STATE_NAME', 'population',
       'lat', 'lon']].reset_index()
        print(data.columns)
        # input()
        

        data = data[(data['lat'] > 25) & (data['lat'] < 49) & (data['lon'] > -124.8) & (data['lon'] < -66.9)]
        data['SUM'] = data['HD'] + data['LOW']


        data=data.merge(data_stores_lines,left_on='index',right_on='store',how='left').reset_index()

        data=data.merge(df_lines,left_on='index',right_on='ini_store',how='left')
        data['best_objective']=best_objective

        data=data.fillna(99999) 

        # data['check'] = data['store'].astype(int).apply(lambda /x: 9000000 if x == -1 else -1)


        print(data.shape)

        data.to_excel(r'H:\My Drive\Advanced_Simulation\HW5\Model2-Q3\HD_LOW.xlsx',index=False)




















        # Display the DataFrame
        
        # print(best_objective)
        # with open('best_objective.txt', 'w') as f:
        #     f.write(best_objective)

        # print(df_lines)
        # df_lines.to_excel('df_lines.xlsx',index=False)
        
        # print(data_stores_lines)
        # data_stores_lines.to_excel('data_stores_lines.xlsx',index=False)




# dataframe_path='gurobi_output30_100.log'
# format_csv(dataframe_path)