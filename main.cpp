#include <iostream>
#include <vector>
#include <queue>
#include <set>
#include <map>
#include <cmath>
#include <memory>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <utility>
#include <limits>
#include <algorithm>
#include <random>


using namespace std;


class FeatureSelection {
    private:
        int NumberOfFeature;
        mt19937 RandomNumber;  //random number generator

    public:
        FeatureSelection(int num) : NumberOfFeature(num), RandomNumber(random_device{}()) {}

        //Random evaluation that just returns number from 0 to 100
        float RandomEvaluationStub(const set<int>& SubsetOfFeatures) {
            uniform_real_distribution<float> dist(0.0f, 100.0f);
            return dist(RandomNumber);
        }

        // Forward Selection
        pair<set<int>, float> forward_selection() {
            set<int> CurrentFeatures;
            float BestScore = RandomEvaluationStub(CurrentFeatures);
            set<int> BestFeatures = CurrentFeatures;

            cout << "Using no features and random evaluation, I get an accuracy of "<< BestScore << "%\n \nBeginning search.\n";


            return {BestFeatures, BestScore};
        }

        // Backward Elimination
        pair<set<int>, float> backward_elimination() {
            set<int> CurrentFeatures;
            for (int i = 1; i <= NumberOfFeature; i++) {
                CurrentFeatures.insert(i);
            }

            float BestScore = RandomEvaluationStub(CurrentFeatures);
            set<int> BestFeatures = CurrentFeatures;

            cout << "Using all features and random evaluation, I get an accuracy of " << BestScore << "%\n \nBeginning search.\n";
            return {BestFeatures, BestScore};
        }

    private:
    
};


int main() {
    cout << "Welcome to the Feature Selection Algorithms.\n";

    int NumberOfFeature;

    cout << "Please enter total number of features: ";
    cin >> NumberOfFeature;

    if (!cin || NumberOfFeature <= 0) {
        cerr << "Features should be greater than 0. Please reenter:\n";
        return 1;
    }

    FeatureSelection feature_selection(NumberOfFeature);

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
            feature_selection.forward_selection();
            break;

        case 2:
            cout << "\nBackward Elimination:\n";
            feature_selection.backward_elimination();
            break;

        default:
            cout << "Please put number between 1 and 2\n";
    }

    return 0;
}
