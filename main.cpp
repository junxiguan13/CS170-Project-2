#include <iostream>
#include <vector>
#include <random>
#include <unordered_set>

using namespace std;

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<double> dist(0.0, 100.0);

double state_prec (const vector<int>& curr_state) {
    double x = dist(gen);//generate a number between 0 and 100
    return x;
}

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

    cout << "Using no features and \"random\" evaluation, I get an accuracy of ";
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
    cout << "} and \"random\" evaluation, I get an accuracy of ";
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

    int NumberOfFeature;

    cout << "Please enter total number of features: ";
    cin >> NumberOfFeature;

    if (!cin || NumberOfFeature <= 0) {
        cerr << "Features should be greater than 0. Please reenter:\n";
        return 1;
    }

    vector<int> init_state;
    for (int i = 1; i <= NumberOfFeature; ++i) {
        init_state.push_back(i);
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