#include <bits/stdc++.h> ///<!---Used this header file got lazy importing all individual files for this project---!>


using namespace std;

// Define a structure to represent a rule
struct Rule {
    string conclusion;
    vector<string> premises;
};

// Define a function to perform forward chaining
void forwardChaining(unordered_map<string, bool>& facts, vector<Rule>& rules) {
    bool newFact = true;
    while (newFact) {
        newFact = false;
        for (Rule& rule : rules) {
            if (facts[rule.conclusion]) {
                // If the conclusion is already known, skip this rule
                continue;
            }
            bool allPremisesTrue = true;
            for (string& premise : rule.premises) {
                if (!facts[premise]) {
                    // If any premise is false, skip this rule
                    allPremisesTrue = false;
                    break;
                }
            }
            if (allPremisesTrue) {
                // If all premises are true, add the conclusion as a new fact
                facts[rule.conclusion] = true;
                newFact = true;
            }
        }
    }
}

int main() {
    // Define some initial facts
    unordered_map<string, bool> facts = {
        {"A", true},
        {"B", true},
        {"C", false},
        {"D", false},
    };
    
    // Define some rules
    vector<Rule> rules = {
        {"E", {"A", "B"}},
        {"F", {"C", "D"}},
        {"G", {"E", "F"}},
    };
    
    // Perform forward chaining to derive new facts
    forwardChaining(facts, rules);
    
    // Print out the final set of facts
    cout << "Final set of facts:\n";
    for (auto& [fact, value] : facts) {
        cout << fact << ": " << value << "\n";
    }
    
    return 0;
}