#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <ctime>
#include <cstdlib>
#include <algorithm> // To Include the 'transform' function

using namespace std;

// Helper function to choose a random element from a vector
template<typename T>
T random_element(const vector<T>& vec) {
    if (vec.empty()) {
        throw runtime_error("Cannot choose a random element from an empty vector.");
    }
    return vec[rand() % vec.size()];
}

class Eliza {
public:
    Eliza() {
        srand(time(0));
        initialize_patterns();
    }

    void run() {
        cout << "Eliza: Hello! How can I help you today?" << endl;
        while (true) {
            string user_input;
            cout << "You: ";
            getline(cin, user_input);

            if (user_input.empty()) {
                cout << "Eliza: Goodbye!" << endl;
                break;
            }

            bool responseGenerated = false; // To track if a response has been generated

            // Iterate over symbolic logic and associated words to find a match
            for (const auto& entry : symbolicLogicToWords) {
                for (const string& word : entry.second) {
                    if (contains_word(user_input, word)) {
                        cout << "Eliza: " << patterns[entry.first] << endl;
                        responseGenerated = true;
                        break;
                    }
                }
                if (responseGenerated) {
                    break;
                }
            }

            if (!responseGenerated) {
                cout << "Eliza: I'm not sure I understand. Can you please rephrase that?" << endl;
            }
        }
    }

private:
    map<string, string> patterns;  // Store patterns and responses as symbolic notation
    map<string, vector<string>> symbolicLogicToWords; // Map symbolic logic to words

    void initialize_patterns() {
        // Add patterns and responses here
        patterns["Greeting"] = "Hello! What's up?";
        patterns["Inquiry"] = "I'm just a computer program, so I don't have feelings, but I'm here to assist you.";
        patterns["Farewell"] = "Goodbye! If you have more questions, feel free to ask.";
        patterns["Identity"] = "You can call me Eliza.";
        patterns["Assistance"] = "I'm here to listen and provide responses. How can I assist you?";
        patterns["Appreciation"] = "You're welcome! If you need any more help, just ask.";
        patterns["Emotion"] = "I'm sorry, what is your problem?";

        // Map symbolic logic to words
        symbolicLogicToWords["Greeting"] = {"hello", "hi", "hey"};
        symbolicLogicToWords["Inquiry"] = {"how are you", "how do you do"};
        symbolicLogicToWords["Farewell"] = {"goodbye", "bye"};
        symbolicLogicToWords["Identity"] = {"name", "who are you"};
        symbolicLogicToWords["Assistance"] = {"help", "support"};
        symbolicLogicToWords["Appreciation"] = {"thanks", "thank you"};
        symbolicLogicToWords["Emotion"] = {"sad", "happy", "confused", "shy"};
    }

    bool contains_word(const string& input, const string& word) {
        string lower_input = to_lower(input);
        string lower_word = to_lower(word);
        return lower_input.find(lower_word) != string::npos;
    }

    string to_lower(const string& str) {
        string lower_str = str;
        transform(lower_str.begin(), lower_str.end(), lower_str.begin(), ::tolower);
        return lower_str;
    }
};

int main() {
    Eliza eliza;
    eliza.run();
    return 0;
}