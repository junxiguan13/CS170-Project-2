#include <iostream>
#include <vector>
#include <random>
#include <unordered_set>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>

using namespace std;

//struct variables and global variable
struct Instance {
    int label;//class label
    vector<double> features;
};
vector<Instance> dataset;//store all data
int num_features;//store the number of features

//global functions
void load_and_normalize(const string& filename) {
    //first reset the global variable and the old data
    dataset.clear();

    //if no features input
    if (num_features <= 0) {
        cout << "Error! The number of features has not been set." << endl;
        exit(1);
    }
    //if can not open file
    ifstream fin(filename);
    if (!fin) {
        cout << "Error! Can not open file " << filename << endl;
        exit(1);
    }

    //reading input from the file
    double label_val;//we assume only using 1 or 2 as label
    while (fin >> label_val) {
        Instance inst;
        inst.label = static_cast<int>(label_val);//notes that label_val(file data) is double type
        
        for (int i = 0; i < num_features; ++i) {//set all features; NOTE THAT num_features must be known
            double feat_val;
            if (!(fin >> feat_val)) {//check if # of features is correct
                cout << "Error! Not enough features." << endl;
                exit(1);
            }

            inst.features.push_back(feat_val);
        }//end for loop, got all features

        dataset.push_back(inst);//push the inst type variable to the entire dataset; ready to train and test
    }//end while loop; stop reading file

    //we need mean, std(standard deviation) to normalize the data
    int num_instances = dataset.size();//how many instances in the dataset
    if (num_instances == 0) {
        cout << "Error! No input from the file " << filename << endl;
        exit(1);
    }
    
    for (int i = 0; i < num_features; ++i) {// to compute the mean and std of EACH FEATURE one by one
        double mean_sum = 0;
        for (int j = 0; j < num_instances; ++j) {
            mean_sum += dataset[j].features[i];
        }
        double feature_mean = mean_sum / num_instances;//got the mean

        double std_sum = 0;
        for (int j = 0; j < num_instances; ++j) {
            double var_diff = dataset[j].features[i] - feature_mean;
            std_sum += var_diff * var_diff;
        }
        double feature_std = sqrt(std_sum / num_instances);//got the std
        if (feature_std == 0) {//check if std is zero
            feature_std = 1;
        }

        //rewrite the data with normalization
        for (int j = 0; j < num_instances; ++j) {
            dataset[j].features[i] = (dataset[j].features[i] - feature_mean) / feature_std;
        }
    }//end for loop(normalization)
}

int nn_classifier(int test_index, const vector<int>& feat_subset) {
    //we need one test data and the rest are train data
    int num_instances = dataset.size();
    const Instance& test_inst = dataset[test_index];//got the test data

    double best_dist = -1;
    int best_label = -1;
    for (int i = 0; i < num_instances; ++i) {//find the train data
        if (i == test_index) {//do not use the test data itself as train data
            continue;
        }

        const Instance& train_inst = dataset[i];
        double Euc_sum = 0;
        int num_subsetFeat = feat_subset.size();
        for (int j = 0; j < num_subsetFeat; ++j) {//find the sum of Euc dist for all features
            int feat_index = feat_subset[j] - 1;
            double Euc_diff = train_inst.features[feat_index] - test_inst.features[feat_index];
            Euc_sum += Euc_diff * Euc_diff;
        }

        if (best_dist == -1 || Euc_sum < best_dist) {//if best dist is default or it is larger than the euc dist square
            best_dist = Euc_sum;//update dist for the current set
            best_label = train_inst.label;//update label for the current set
        }
    }//end for loop; found the best label for the current traning SET

    return best_label;
}

double leave_one_out_accuracy(const vector<int>& feat_subset) {//check two cases: one is when using no features; or with features using
    int num_instances = dataset.size();

    //case #1: no features, use default rate
    if (feat_subset.empty()) {
        int count_label1 = 0;
        int count_label2 = 0;

        for (int i = 0; i < num_instances; ++i) {
            if (dataset[i].label == 1) {
                ++count_label1;
            }
            else {
                ++count_label2;
            }
        }//end for loop(counting # of label 1 and 2)

        if (count_label2 > count_label1) {
            return (static_cast<double>(count_label2) / num_instances * 100);
        }
        else {
            return (static_cast<double>(count_label1) / num_instances * 100);
        }
    }//case #1 end

    //case #2: using features, which is the normal case
    //run the nn_classifier to determine the accuracy of the model
    int correct_labels = 0;
    for (int i = 0; i < num_instances; ++i) {
        int predicted_label = nn_classifier(i, feat_subset);
        if (predicted_label == dataset[i].label) {
            ++correct_labels;
        }
    }

    double accu_rate = static_cast<double>(correct_labels) / num_instances * 100;
    return accu_rate;
}

double state_prec (const vector<int>& curr_state) {
    return leave_one_out_accuracy(curr_state);
}

/*this is the old prec function(random) and relative variables
random_device rd;
mt19937 gen(rd());
uniform_real_distribution<double> dist(0.0, 100.0);

double state_prec (const vector<int>& curr_state) {
    double x = dist(gen);//generate a number between 0 and 100
    return x;
}*/

void greedy_forward_selection (const vector<int>& all_feat) {
    const int num_feat = all_feat.size();
    //size of the feature vector

    vector<int> best_state;
    //to store the best state, empty as default
    vector<int> curr_state;
    //to store the current state

    unordered_set<int> used_feat;
    used_feat.reserve(num_feat);
    //using hash table to check if a feature has been already used

    double curr_prec = state_prec(best_state);

    cout << "Using no features and leave-one-out validation method, I get an accuracy of ";
    cout << curr_prec;
    cout << "% " << endl;

    cout << endl << "Beginning search." << endl;    

    for (int i = 0; i < num_feat; ++i) {//will only add num_feat features to maximum
        int best_feat;
        //will store the best feature on this level
        double best_prec = -1.0;
        //will store the corresponding precision of the best state on this level
        bool better_feat = false;
        //check if this level finds any feature to add

        for (int j = 0; j < num_feat; ++j) {//find the best feature on this level
            int curr_feat = all_feat[j];
            //going over all features

            //if this feature is already added to the current state, skip it
            if (used_feat.count(curr_feat)) {
                continue;
            }

            //temporarily add this feature to the current state
            curr_state.push_back(curr_feat);
            double this_prec = state_prec(curr_state);//prec of the temp state

            //print the state and state precision
            cout << "Using feature(s) { ";
            for (int k = 0; k < curr_state.size(); ++k) {
                cout << curr_state.at(k) << " ";
            }
            cout << "} accuracy is " << this_prec << "%" << endl;
            //print done

            //update best feature on this level
            if (!better_feat || best_prec < this_prec) {//if no feature has been added yet or if this level's precisioni is larger, then update
                best_feat = curr_feat;
                best_prec = this_prec;
                better_feat = true;
            }

            //remove the temporarily added feature, restore curr_state
            curr_state.pop_back();
        }//j loop ended

        //if this level could not find any new feature to add, stop the search
        if (!better_feat) {
            break;
        }

        //permanently add the best feature of this level, update hash table
        curr_state.push_back(best_feat);
        used_feat.insert(best_feat);

        cout << "Feature set { ";
        for (int k = 0; k < curr_state.size(); ++k) {
            cout << curr_state.at(k) << " ";
        }
        cout << "} was best, accuracy is " << best_prec << "%" << endl;

        //compare this level's best with overall best
        if (best_prec < curr_prec) {
            cout << "(Warning, Accuracy has decreased!)" << endl;
        } else {
            curr_prec  = best_prec;
            best_state = curr_state;
        }
    }//i loop ended

    cout << "Finished search!! The best feature subset is { ";
    for (int i = 0; i < best_state.size(); ++i) {
        cout << best_state.at(i) << " ";
    }
    cout << "} which has an accuracy of " << curr_prec << "%" << endl;
}//function ended

void greedy_backward_selection (const vector<int>& all_feat) {
    // start from the full set of features
    vector<int> curr_state = all_feat;
    // current state (we will remove one feature each level)

    vector<int> best_state = curr_state;
    // to store the best state overall

    double curr_prec = state_prec(curr_state);
    // Evaluate full set

    cout << "Using all features { ";
    for (int i = 0; i < all_feat.size(); ++i) {
        cout << all_feat.at(i) << " ";
    }
    cout << "} and leave-one-out validation method, I get an accuracy of ";
    cout << curr_prec;
    cout << "% " << endl;

    cout << "Beginning search." << endl << endl;

    // Continue until only one feature remains
    while (curr_state.size() > 1) {
        double best_prec = -1.0;
        // best accuracy ON THIS LEVEL
        vector<int> best_state_level;
        // best feature subset on this level

        // Try removing each feature one by one
        for (int i = 0; i < curr_state.size(); ++i) {
            // make a temporary copy
            vector<int> temp_state = curr_state;
            // remove the i-th feature
            temp_state.erase(temp_state.begin() + i);

            double this_prec = state_prec(temp_state);

            // Print temp set with correct formatting
            cout << "Using feature(s) { ";
            for (int k = 0; k < temp_state.size(); ++k) {
                cout << temp_state.at(k) << " ";
            }
            cout << "} accuracy is " << this_prec << "%" << endl;

            // Track best removal
            if (this_prec > best_prec) {
                best_prec = this_prec;
                best_state_level = temp_state;
            }
        } // end for i

        // Print the best set for this level
        cout << "Feature set { ";
        for (int k = 0; k < best_state_level.size(); ++k) {
            cout << best_state_level.at(k) << " ";
        }
        cout << "} was best, accuracy is " << best_prec << "%" << endl;

        // Check if accuracy decreased
        if (best_prec < curr_prec) {
            cout << "(Warning, Accuracy has decreased!)" << endl;
        } else {
            curr_prec  = best_prec;//update the best prec ON THIS LEVEL to curr_prec
            best_state = best_state_level;
        }

        // Update the current set to the best one
        curr_state = best_state_level;

        cout << endl;
    } // end while

    // Print final result
    cout << "Finished search!! The best feature subset is { ";
    for (int i = 0; i < best_state.size(); ++i) {
        cout << best_state.at(i) << " ";
    }
    cout << "} which has an accuracy of " << curr_prec << "%" << endl;
}

int main() {
    cout << "Welcome to the Feature Selection Algorithms.\n";

    cout << "Please enter the filename to read inputs. You can use the default test files \'small-test-dataset-2-2.txt\' and \'large-test-dataset-2.txt\'." << endl;
    string filename;
    getline(cin, filename);

    int NumberOfFeature;
    cout << "Please enter total number of features." << endl;
    cout << "For default \'small-test-dataset-2-2.txt\', use features = 10." << endl;
    cout << "For default \'large-test-dataset-2.txt\', use features = 40." << endl;
    cin >> NumberOfFeature;
    num_features = NumberOfFeature;
    /*
    Small Dataset: small-test-dataset.txt (Has 100 instances and 10 features)
    Large Dataset: large-test-dataset.txt (Has 1000 instances, and 40 features)
    */
    if (!cin || NumberOfFeature <= 0) {
        cerr << "Features should be greater than 0. Please reenter:\n";
        return 1;
    }
    vector<int> init_state;
    for (int i = 1; i <= NumberOfFeature; ++i) {
        init_state.push_back(i);
    }
    
    load_and_normalize(filename);

    //test small cases
    cout << "If using \'small-test-dataset-2-2.txt\', do you wish to check a qucik test? (Only with features {3, 5, 7}, which should give an accuracy of 0.89). If yes, please enter \'1\'. Otherwwise, please enter \'0\'." << endl;
    bool quicktest1;
    cin >> quicktest1;
    if (quicktest1) {
        vector<int> subset1 = {3, 5, 7};
        double accu1 = leave_one_out_accuracy(subset1);
        cout << "Using features {3, 5, 7}, the accuracy is " << accu1 << "%" << endl;
        return 0;
    }
    cout << "If using \'large-test-dataset-2.txt\', do you wish to check a qucik test? (Only with features {1, 15, 27}, which should give an accuracy of 0.949.) If yes, please enter \'1\'. Otherwwise, please enter \'0\'." << endl;
    bool quicktest2;
    cin >> quicktest2;
    if (quicktest2) {
        vector<int> subset2 = {1, 15, 27};
        double accu2 = leave_one_out_accuracy(subset2);
        cout << "Using features {1, 15, 27}, the accuracy is " << accu2 << "%" << endl;
        return 0;
    }

    // Menu
    cout << "\nType the number of the algorithm you want to run.\n";
    cout << "1. Forward Selection\n";
    cout << "2. Backward Elimination\n";

    int choice;
    cout << "Your choice: ";
    cin >> choice;

    if (!cin) {
        cerr << "Try again with number between 1 and 2.\n";
        return 1;
    }

    switch (choice) {
        case 1:
            cout << "\nForward Selection:\n";
            greedy_forward_selection (init_state);
            break;

        case 2:
            cout << "\nBackward Elimination:\n";
            greedy_backward_selection (init_state);
            break;

        default:
            cout << "Please put number between 1 and 2\n";
    }

    return 0;
}
