#include <iostream>
#include <vector>
#include <random>
#include <unordered_set>

using namespace std;

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<double> dist(0.0, 100.0);

double state_prec (const vector<char>& curr_state) {
    double x = dist(gen);//generate a number between 0 and 100
    return x;
}

void greedy_forward_selection (const vector<char>& all_feat) {
    const int num_feat = all_feat.size();
    //size of the feature vector

    vector<char> best_state;
    //to store the best state, empty as default
    vector<char> curr_state;
    //to store the current state

    unordered_set<char> used_feat;
    used_feat.reserve(num_feat);
    //using hash table to check if a feature has been already used

    double curr_prec = state_prec(best_state);

    cout << "Using no features and \"random\" evaluation, I get an accuracy of ";
    cout << curr_prec;
    cout << "% " << endl;

    cout << endl << "Beginning search." << endl;    

    for (int i = 0; i < num_feat; ++i) {//will only add num_feat features to maximum
        char best_feat;
        //will store the best feature on this level
        double best_prec = -1.0;
        //will store the corresponding precision of the best state on this level
        bool better_feat = false;
        //check if this level finds any feature to add

        for (int j = 0; j < num_feat; ++j) {//find the best feature on this level
            char curr_feat = all_feat[j];
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


int main() {
    vector<char> all_feat;
    all_feat.push_back('1');
    all_feat.push_back('2');
    all_feat.push_back('3');
    all_feat.push_back('4');

    greedy_forward_selection(all_feat);

    return 0;
}