__author__ = "katie"
__date__ = "$Mar 8, 2017 1:13:12 PM$"

import math
import random 

# This program implements a Naive Bayesian classifier on census data
# in an attempt to classify individuals' income based on the attributes 
# seen below

# age: continuous.
# workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# fnlwgt: continuous.
# education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# education-num: continuous.
# marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# sex: Female, Male.
# capital-gain: continuous.
# capital-loss: continuous.
# hours-per-week: continuous.
# native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

# Missing values are strategically accounted for in 2 ways and 
# continuous attributes are both binned and assumed to be Gaussian 
# in order to compare the many available methods

# K-fold cross validation is performed as validation of the algorithm 

# This method extracts the following information about the raw data
# the records that contain missing values,
# the minimum of each continuous attribute
# the mean of each continuous attribute
def preprocess_data(lines):
    
    # Define categorical attribute values (only those that have missing values are needed)  
    workclasses = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
    occupations = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
    countries = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
    
    # Determine total number of records
    num_records = len(lines)
            
    # Create lists of indices for records with missing values
    missing_workclass = []
    missing_occupation = []
    missing_country = []
    
    # Initialize min variables
    age_min = 10001000000000
    fnlwgt_min = 1000000000
    education_num_min = 1000000000
    capital_gain_min = 1000000000
    capital_loss_min = 1000000000
    hours_per_week_min = 1000000000
    
    # Initialize means list
    means = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # Create list of index values for each continuous attribute
    cont_attr_indices = [0, 2, 4, 10, 11, 12]
    
    # Create a list to hold the data (from csv lines-->list format)
    data = []
    
    # Loop through lines of data and save lines with missing attribute values
    for i in range(0, len(lines)): 
        
        #  Split csv form by commas
        attribute_values = lines[i].split(',')
        
        # Check if line is empty
        if len(attribute_values) != 1:
            
            # Strip blank spaces from beginning and end
            for j in range(0, len(attribute_values)):
                attribute_values[j] = attribute_values[j].strip(' ')
                
                # Calculate a sum to be used for mean of each continuous attributes
                if j in cont_attr_indices: 
                    means[j] += int(attribute_values[j])
                
            # Add to data list
            data.append(attribute_values)
            
            # Check if entries are missing and add to respective lists
            if attribute_values[1] not in workclasses:
                missing_workclass.append(i)
            if attribute_values[6] not in occupations:
                missing_occupation.append(i)
            if attribute_values[13] not in countries:
                missing_country.append(i)
            
            # Find each continuous attribute's min
            if int(attribute_values[0]) < age_min:
                age_min = int(attribute_values[0])
            if int(attribute_values[2]) < fnlwgt_min:
                fnlwgt_min = int(attribute_values[2])
            if int(attribute_values[4]) < education_num_min:
                education_num_min = int(attribute_values[4])
            if int(attribute_values[10]) < capital_gain_min:
                capital_gain_min = int(attribute_values[10])
            if int(attribute_values[11]) < capital_loss_min:
                capital_loss_min = int(attribute_values[11])
            if int(attribute_values[12]) < hours_per_week_min:
                hours_per_week_min = int(attribute_values[12])
    
    # Calculate mins and means for continuous attribute handling
    mins = [age_min, fnlwgt_min, education_num_min, capital_gain_min, capital_loss_min, hours_per_week_min]
    for i in range(0, len(means)):
        means[i] = means[i]/(num_records)
        
    return [data, missing_workclass, missing_occupation, missing_country, mins, means]

# This method creates a new data list, excluding all records with missing values
def remove_missing_value_records(data, missing_workclass, missing_occupation, missing_country):

    # Initialize empty list of data
    incomplete_data = []
    
    # Loop through data
    for i in range(0, len(data)):
        
        # Check if record contains a missing value
        if (i not in missing_workclass) and (i not in missing_occupation) and (i not in missing_country):
            
            # If not, append to list
            incomplete_data.append(data[i])
            
    return incomplete_data

# This method determines the attribute value for a given attribute with the highest probability
def value_max_prob(attr_dict):
    
    max_val = -1        # Maximum probability found so far
    max_key = ''        # Key (attribute value) of largest probability 
    
    # Loop through each attribute value 
    for key in attr_dict:
        
        # Determine if probability is maximum
        if attr_dict[key] > max_val:
            
            # If so, save as new attribute value with maximum probability
            max_val = attr_dict[key]
            max_key = key

    return max_key

# This method creates a new data list, including records with missing values
# and adjusting those missing values to reflect the most common attribute value 
# given the particular income class the record falls into
def fill_in_missing_values(data, missing_workclass, missing_occupation, missing_country, model1):
    
    # Access the large and small income probability dictionaries
    attr_prob_large_income = model1[0]
    attr_prob_small_income = model1[1]
    
    # Initialize empty list of data
    complete_data = []
    
    # Loop through data 
    for i in range(0, len(data)):
        
        # Access current record
        record = data[i]
        
        # Check if this record contains a missing workclass value
        if (i in missing_workclass):
            
            # If so, check what income class the record falls into
            # and substitute the attribute value with the highest probability
            # for the given class
            if record[len(record)-1] == '<=50K\n':
                new_value = value_max_prob(attr_prob_large_income[1])
            else:
                new_value = value_max_prob(attr_prob_small_income[1])
            record[1] = new_value
        
        # Check if this record contains a missing occupation value
        if (i in missing_occupation):
            
            # If so, check what income class the record falls into
            # and substitute the attribute value with the highest probability
            # for the given class
            if record[len(record)-1] == '<=50K\n':
                new_value = value_max_prob(attr_prob_large_income[6])
            else:
                new_value = value_max_prob(attr_prob_small_income[6])
            record[6] = new_value
        
        # Check if this record contains a missing country value
        if (i in missing_country):
            
            # If so, check what income class the record falls into
            # and substitute the attribute value with the highest probability
            # for the given class
            if record[len(record)-1] == '<=50K\n':
                new_value = value_max_prob(attr_prob_large_income[13])
            else:
                new_value = value_max_prob(attr_prob_small_income[13])
            record[13] = new_value
        
        # Append updated record to data list
        complete_data.append(record)
    
    return complete_data

# Method that divides all continuous variables into equal width bins
# Bin width is selected for each attribute based on data visualization analysis
def equal_width_binning(data, mins):
    
    # Select appropriate width values for each attribute
    cont_attr_width = [5, 30000, 2, 3000, 250, 6] 
    
    # Create list of index values for each continuous attribute
    cont_attr_indices = [0, 2, 4, 10, 11, 12]
    
    # Loop through each line of data
    k = 0
    for data_pt in data:
        
        # Loop through all continuous attributes
        j = 0
        for i in cont_attr_indices:
            
            # Access current value of a given continous attribute
            attr_val = int(data_pt[i])
            
            # Calculate the width based on the attribute
            width = cont_attr_width[j]
            
            # Recalculate attribute value based on appropriate bin 
            #This rounds up so bin with value x contains [x-(width/2), x+(width/2))
            data_pt[i] = (math.floor((attr_val - mins[j])/width)*width) + mins[j] + (width/2)
            
            j += 1
        
        # Reassign data point to data list
        data[k] = data_pt
        
        k += 1
        
    return data

# Method that trains the naive bayesian classifier by counting the number of 
# occurences of each attribute in order to achieve the values required 
# by the numerator of Bayes' Theorem
def train_naive_classifier(data, means, assumeGaussian):

    # Create list of dictionaries to store count of each attribute value for large and small incomes
    # Each element of the list is a dictionary which corresponds to an attribute
    # Each key of each dictionary corresponds to a particular value for the given attribute
    # Each value of each dictionary corresponds to the count of occurences of that attribute value 
    attr_count_large_income = [{},{},{},{},{},{},{},{},{},{},{},{},{},{}]
    attr_count_small_income = [{},{},{},{},{},{},{},{},{},{},{},{},{},{}]
    
    # Count total number of large incomes and total number of small incomes
    num_large_incomes = 0
    num_small_incomes = 0
    
    # Track indices of continuous attributes (not a concern if already binned)
    cont_attr_indices = []
    if assumeGaussian: 
        cont_attr_indices = [0, 2, 4, 10, 11, 12]
    
    # Hard coded standard deviation values from Tableau observations
    std = [13.64, 0.0, 105550.0, 0.0, 2.573, 0.0, 0.0, 0.0, 0.0, 0.0, 7385.0, 403.0, 12.35, 0.0]
    
    # Loop through every line of the data
    for line in data:
        
        # Determine if this individual falls into the large or small income category
        if line[len(line)-1] == ">50K\n":
            
            # If large, increment the count of individuals with large incomes
            num_large_incomes += 1
            
            # Loop through remaining attributes for this particular individual 
            # (not the income attribute)
            for i in range(0, len(line)-1):
                
                attribute = line[i]

                # Access the existing dictionary of counts for that attribute 
                attr_dict = attr_count_large_income[i]

                # If the dictionary does not yet have an entry for the particular 
                # attribute value, then add it and start the count at 2 
                # (since we are adding 1 to avoid cases when the model does not have a value)
                if attribute not in attr_dict: 
                    attr_dict[attribute] = 2
                
                # Otherwise increment the count for that attribute value
                else: 
                    attr_dict[attribute] += 1
            
                # Return the edited dictionary to the list of all attribute dictionaries
                attr_count_large_income[i] = attr_dict
        else:
            
            # If small, increment the count of individuals with small incomes
            num_small_incomes += 1
            
            # This loop does the same function as the for loop above just for small incomes 
            # (see above for thorough documentation)
            for i in range(0, len(line)-1):
                attribute = line[i]
                
                attr_dict = attr_count_small_income[i]
                
                if attribute not in attr_dict:
                    attr_dict[attribute] = 2
                else: 
                    attr_dict[attribute] += 1
                
                attr_count_small_income[i] = attr_dict
    
    # Turn all counts of large income into probabilities (divide by total # large or small incomes)
    attr_prob_large_income = [{},{},{},{},{},{},{},{},{},{},{},{},{},{}]
    attr_prob_small_income = [{},{},{},{},{},{},{},{},{},{},{},{},{},{}]
    i = 0    
    for attr in attr_count_large_income:
        
        # Check if dictionary is empty
        if bool(attr):
            
            # If not, for each key value, convert to a probability
            for key in attr:
                attr[key] = float(attr[key])/float(num_large_incomes)
                
            # Reassign updated dictionary
            attr_prob_large_income[i] = attr
        i += 1
        
    # Turn all counts of small income into probabilities (same as above)
    i = 0
    for attr in attr_count_small_income:
        if bool(attr):
            for key in attr:
                attr[key] = float(attr[key])/float(num_small_incomes)
            attr_prob_small_income[i] = attr
        i += 1
    
    # If we are assuming continuous attributes to be Gaussian
    if assumeGaussian: 
        
        # Recalculate probability of continuous attributes by assuming 
        # Gaussian distribution
        for index in cont_attr_indices:
            
            # Replace large income probabilities for each continuous attribute with normal pdf 
            dict1 = attr_prob_large_income[index]
            for key in dict1:
            
                # Calculate y value of normal distribution based on x value (value of attribute)
                prob = (1/(math.sqrt(2*math.pi)*std[index]))*math.pow(math.e, -(math.pow(key-means[index],2)/(2*math.pow(std[index],2))))
               
                # Reassign probability 
                dict1[key] = prob
                
            # Reassign dictionary
            attr_prob_large_income[index] = dict1
            
            # Replace small income probabilities for each continuous attribute with normal pdf (same as above)
            dict2 = attr_prob_small_income[index]
            for key in dict2:
                prob = (1/(math.sqrt(2*math.pi)*std[index]))*math.pow(math.e, -(math.pow(key-means[index],2)/(2*math.pow(std[index],2))))
                dict2[key] = prob
            attr_prob_small_income[index] = dict2
            
    # Return the probability of each attribute value for each income class and P(income>50k) and P(income<=50k)   
    return [attr_prob_large_income, attr_prob_small_income, float(num_large_incomes)/float(num_small_incomes+num_large_incomes), float(num_small_incomes)/float(num_small_incomes+num_large_incomes), num_large_incomes, num_small_incomes]

# This method measures the performance of a trained model given test data, returning:
# accuracy
# precision
# recall
def evaluate_model(model, testing_data):
    
    # Recall the Bayes' theorem numerator...
    # P(income>50k | age=_, workclass=_, etc.) = P(age=_ | income>50k)P(workclass=_ | income>50k)...P(income>50k)
    # P(income<=50k | age=_, workclass=_, etc.) = P(age=_ | income<=50k)P(workclass=_ | income<=50k)...P(income<=50k)
    
    # Access all aspects of the trained model
    large_income_attr_list = model[0]
    small_income_attr_list = model[1]
    prob_large_income = model[2]
    prob_small_income = model[3]     
    num_large_incomes = model[4] 
    num_small_incomes = model[5]
    
    # Intialize confusion matrix values
    # Here we are assuming "positive" is >50K and "negative" is <=50K
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    # Loop through every test data point
    for data_pt in testing_data:
        
        # Keep running product of Bayes' theorem numerator for both classes
        large_income_numerator = 1.0
        small_income_numerator = 1.0
        
        # Loop through each attribute value of that data point
        for i in range(0, len(data_pt)-1):
            
            # Access the attribute value
            attr_value = data_pt[i]
            
            # Access the attribute's corresponding probability dictionary 
            large_income_attr_dict = large_income_attr_list[i]
            small_income_attr_dict = small_income_attr_list[i]

            # If this value exists in the large income dictionary, multiply the corresponding probability
            if attr_value in large_income_attr_dict:
                large_income_numerator = large_income_numerator * large_income_attr_dict[attr_value]
            
            # Otherwise, multiply 1/num_large_incomes to get as close to zero as possible
            else:
                large_income_numerator = large_income_numerator * float(1.0/num_large_incomes)
            
            # If this value exists in the small income dictionary, multiply the corresponding probability
            if attr_value in small_income_attr_dict:
                small_income_numerator = small_income_numerator * small_income_attr_dict[attr_value]
            
            # Otherwise, multiply 1/num_small_incomes to get as close to zero as possible
            else:
                small_income_numerator = small_income_numerator * float(1.0/num_small_incomes)

        # Finally, multiply by probability of each income class
        large_income_numerator = large_income_numerator * prob_large_income
        small_income_numerator = small_income_numerator * prob_small_income

        # Count number of tp, fp, tn, fn where "positive" = >50K and "negative" = <=50K
        if large_income_numerator > small_income_numerator and data_pt[14] == ">50K\n":
            tp += 1
        elif large_income_numerator > small_income_numerator and data_pt[14] == "<=50K\n":
            fp += 1
        elif large_income_numerator <= small_income_numerator and data_pt[14] == "<=50K\n":
            tn += 1
        elif large_income_numerator <= small_income_numerator and data_pt[14] == ">50K\n":
            fn += 1
            
    # Calculate accuracy, precision and recall
    acc = float(tp+tn)/float(tp+tn+fp+fn)
    prec = float(tp)/float(tp+fp)
    rec = float(tp)/float(tp+fn)
        
    return [acc, prec, rec]
        
if __name__ == "__main__":
    
    ############################################################
    #            Access data from census.data file             #
    ############################################################
    with open('census.data', 'r') as f:
        lines = f.readlines()
        
    ############################################################
    #        Preprocess data by calculating necessary values   #
    ############################################################
        
    preprocessed_data = preprocess_data(lines)
    data = preprocessed_data[0]
    missing_workclass = preprocessed_data[1]
    missing_occupation = preprocessed_data[2]
    missing_country = preprocessed_data[3]
    mins = preprocessed_data[4]
    means = preprocessed_data[5]
      
    ############################################################
    #        Handle missing values by ignoring records         #
    ############################################################
    
    # Remove records with missing values
    incomplete_data = remove_missing_value_records(data, missing_workclass, missing_occupation, missing_country)
    
    ############################################################
    #     Handle cont. attributes by equal width binning       #
    ############################################################
    
    # Bin incomplete data
    incomplete_data_binned = equal_width_binning(incomplete_data, mins)
    
    # NOTE: assuming Gaussian is done with the 'True' or 'False' argument of train_naive_classifier
    
    #############################################################
    #    Train Naive Bayesian Classifier on incomplete data     #
    #############################################################
    
    # Use this model for handling missing values based on attribute value with highest probability
    model1 = train_naive_classifier(incomplete_data, means, False)
    
    ############################################################
    #        Handle missing values by filling in values        #
    ############################################################
    
    # Fill in missing values by considering value with largest likelihood based on income class
    complete_data = fill_in_missing_values(data, missing_workclass, missing_occupation, missing_country, model1) 
    
    ############################################################
    #     Handle cont. attributes by equal width binning       #
    ############################################################
    
    # Bin complete data
    complete_data_binned = equal_width_binning(complete_data, mins)
    
    # NOTE: assuming Gaussian is done with the 'True' or 'False' argument of train_naive_classifier

    #############################################################
    #         Do K-fold cross validation for all 4 models       #
    #############################################################
    
    #### Ignore records with missing values and use equal width binning ####
    
    # Randomly partition data into 10 partitions
    partitioned_data = [[], [], [], [], [], [], [], [], [], []]
    for data_pt in incomplete_data_binned:
        partitioned_data[random.randint(0,9)].append(data_pt)
    
    # Treat each partition as a test set once
    acc_sum = 0
    prec_sum = 0
    rec_sum = 0
    for i in range(0, 10):
        
        # Isolate testing data
        testing_data = partitioned_data[i]
        
        # Build training dataset
        training_data = []
        for j in range(0, 10):
            if j != i:
                for record in partitioned_data[j]:
                    training_data.append(record)
                    
        # Train naive bayesian classifier on training set
        model = train_naive_classifier(training_data, means, False)
        
        # Calculate accuracy, precision, recall, and F1 measure 
        measures = evaluate_model(model, testing_data)
        
        acc_sum += measures[0]
        prec_sum += measures[1]
        rec_sum += measures[2]
    
    mean_acc = acc_sum/10.0
    mean_prec = prec_sum/10.0
    mean_rec = rec_sum/10.0
    
    print('____________________________________________________________________')
    print('Ignore records with missing values, use equal width binning')
    print('Mean accuracy: ' + str(mean_acc))
    print('Mean precision: ' + str(mean_prec))
    print('Mean recall: ' + str(mean_rec))
    print('Mean F1-measure: ' + str(2*(mean_prec*mean_rec)/(mean_prec+mean_rec)))
    print('____________________________________________________________________')
    print('')
        
    #### Ignore records with missing values and assume cont. attributes are Gaussian ####
    
    # Randomly partition data into 10 partitions
    partitioned_data = [[], [], [], [], [], [], [], [], [], []]
    for data_pt in incomplete_data:
        partitioned_data[random.randint(0,9)].append(data_pt)
    
    # Treat each partition as a test set once
    acc_sum = 0
    prec_sum = 0
    rec_sum = 0
    for i in range(0, 10):
        
        # Isolate testing data
        testing_data = partitioned_data[i]
        
        # Build training dataset
        training_data = []
        for j in range(0, 10):
            if j != i:
                for record in partitioned_data[j]:
                    training_data.append(record)
                    
        # Train naive bayesian classifier on training set
        model = train_naive_classifier(training_data, means, True)
        
        # Calculate accuracy, precision, recall, and F1 measure 
        measures = evaluate_model(model, testing_data)
        
        acc_sum += measures[0]
        prec_sum += measures[1]
        rec_sum += measures[2]
        
    mean_acc = acc_sum/10.0
    mean_prec = prec_sum/10.0
    mean_rec = rec_sum/10.0
    
    print('____________________________________________________________________')
    print('Ignore records with missing values, assume continuous attributes are Gaussian')
    print('Mean accuracy: ' + str(mean_acc))
    print('Mean precision: ' + str(mean_prec))
    print('Mean recall: ' + str(mean_rec))
    print('Mean F1-measure: ' + str(2*(mean_prec*mean_rec)/(mean_prec+mean_rec)))
    print('____________________________________________________________________')
    print('')
    
    #### Fill in missing values and use equal width binning ####
    
    # Randomly partition data into 10 partitions
    partitioned_data = [[], [], [], [], [], [], [], [], [], []]
    for data_pt in complete_data_binned:
        partitioned_data[random.randint(0,9)].append(data_pt)
    
    # Treat each partition as a test set once
    acc_sum = 0
    prec_sum = 0
    rec_sum = 0
    for i in range(0, 10):
        
        # Isolate testing data
        testing_data = partitioned_data[i]
        
        # Build training dataset
        training_data = []
        for j in range(0, 10):
            if j != i:
                for record in partitioned_data[j]:
                    training_data.append(record)
                    
        # Train naive bayesian classifier on training set
        model = train_naive_classifier(training_data, means, False)
        
        # Calculate accuracy, precision, recall, and F1 measure 
        measures = evaluate_model(model, testing_data)
        
        acc_sum += measures[0]
        prec_sum += measures[1]
        rec_sum += measures[2]
    
    mean_acc = acc_sum/10.0
    mean_prec = prec_sum/10.0
    mean_rec = rec_sum/10.0
    
    print('____________________________________________________________________')
    print('Fill in missing values, use equal width binning')
    print('Mean accuracy: ' + str(mean_acc))
    print('Mean precision: ' + str(mean_prec))
    print('Mean recall: ' + str(mean_rec))
    print('Mean F1-measure: ' + str(2*(mean_prec*mean_rec)/(mean_prec+mean_rec)))
    print('____________________________________________________________________')
    print('')
    
    #### Fill in missing values and assume cont. attributes are Gaussian ####
    
    # Randomly partition data into 10 partitions
    partitioned_data = [[], [], [], [], [], [], [], [], [], []]
    for data_pt in complete_data:
        partitioned_data[random.randint(0,9)].append(data_pt)
    
    # Treat each partition as a test set once
    acc_sum = 0
    prec_sum = 0
    rec_sum = 0
    for i in range(0, 10):
        
        # Isolate testing data
        testing_data = partitioned_data[i]
        
        # Build training dataset
        training_data = []
        for j in range(0, 10):
            if j != i:
                for record in partitioned_data[j]:
                    training_data.append(record)
                    
        # Train naive bayesian classifier on training set
        model = train_naive_classifier(training_data, means, True)
        
        # Calculate accuracy, precision, recall, and F1 measure 
        measures = evaluate_model(model, testing_data)
        
        acc_sum += measures[0]
        prec_sum += measures[1]
        rec_sum += measures[2]
        
    mean_acc = acc_sum/10.0
    mean_prec = prec_sum/10.0
    mean_rec = rec_sum/10.0
    
    print('____________________________________________________________________')
    print('Fill in missing values, assume continuous attributes are Gaussian')
    print('Mean accuracy: ' + str(mean_acc))
    print('Mean precision: ' + str(mean_prec))
    print('Mean recall: ' + str(mean_rec))
    print('Mean F1-measure: ' + str(2*(mean_prec*mean_rec)/(mean_prec+mean_rec)))
    print('____________________________________________________________________')
    print('')
    