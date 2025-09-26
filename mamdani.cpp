#include<bits/stdc++.h>
using namespace std;

struct FuzzySet {
    vector<double> points;
};

struct LinguisticVariable {
    string name;
    map<string, FuzzySet> sets;
    double min_range;
    double max_range;
};

struct Rule {
    string speed_is;
    string distance_is;
    string throttle_is;
    string brake_is;
};

double calculateMembership(double value, const FuzzySet& set) {
    if (set.points.size() == 3) {
        double a = set.points[0], b = set.points[1], c = set.points[2];
        if (value <= a || value >= c) return 0.0;
        if (value > a && value <= b) return (value - a) / (b - a);
        if (value > b && value < c) return (c - value) / (c - b);
    }
    else if (set.points.size() == 4) {
        double a = set.points[0], b = set.points[1], c = set.points[2], d = set.points[3];
        if (value <= a || value >= d) return 0.0;
        if (value > a && value < b) return (value - a) / (b - a);
        if (value >= b && value <= c) return 1.0;
        if (value > c && value < d) return (d - value) / (d - c);
    }
    return 0.0;
}

map<string, double> fuzzify(double crispValue, const LinguisticVariable& variable) {
    map<string, double> memberships;
    for (const auto& pair : variable.sets) {
        double membership = calculateMembership(crispValue, pair.second);
        if (membership > 0) {
            memberships[pair.first] = membership;
        }
    }
    return memberships;
}

double defuzzify(const map<string, double>& aggregatedOutputs, const LinguisticVariable& variable) {
    double numerator = 0.0;
    double denominator = 0.0;
    int steps = 100;

    double step_size = (variable.max_range - variable.min_range) / steps;

    for (int i = 0; i <= steps; ++i) {
        double x = variable.min_range + i * step_size;
        double max_membership = 0.0;

        for (const auto& pair : aggregatedOutputs) {
            const string& setName = pair.first;
            double firingStrength = pair.second;
            
            double membership = calculateMembership(x, variable.sets.at(setName));
            max_membership = max(max_membership, min(membership, firingStrength));
        }
        numerator += x * max_membership;
        denominator += max_membership;
    }
    
    return (denominator == 0) ? 0.0 : numerator / denominator;
}

int main() {
    LinguisticVariable speed, distance, throttle, brake;
    speed.name = "Speed";
    speed.min_range = 0;
    speed.max_range = 70;
    speed.sets["Stopped"] = {{ -2, 0, 2 }};
    speed.sets["Very Slow"] = {{ 1, 2.5, 4 }};
    speed.sets["Slow"] = {{ 2.5, 6.5, 10.5 }};
    speed.sets["Medium Fast"] = {{ 6.5, 26.5, 46.5 }};
    speed.sets["Fast"] = {{ 26.5, 70, 70, 70 }};
 
    distance.name = "Distance";
    distance.min_range = 0;
    distance.max_range = 3200;
    distance.sets["At"] = {{ -2, 0, 2 }};
    distance.sets["Very Near"] = {{ 1, 3, 5 }};
    distance.sets["Near"] = {{ 10, 105, 200 }};
    distance.sets["Medium Far"] = {{ 100, 1550, 3200 }};
    distance.sets["Far"] = {{ 1500, 3200, 3200, 3200}};

    throttle.name = "Throttle";
    throttle.min_range = 0;
    throttle.max_range = 100;
    throttle.sets["No"] = {{ -2, 0, 2 }};
    throttle.sets["Very Slow"] = {{ 1, 3, 5 }};
    throttle.sets["Slow"] = {{ 10, 20, 30 }};
    throttle.sets["Medium"] = {{ 20, 50, 80 }};
    throttle.sets["Full"] = {{ 60, 100, 100, 100 }};

    brake.name = "Brake";
    brake.min_range = 0;
    brake.max_range = 100;
    brake.sets["No"] = {{ 0, 0, 0, 40 }};
    brake.sets["Very Slow"] = {{ 20, 50, 80 }};
    brake.sets["Slow"] = {{ 75, 85, 95 }};
    brake.sets["Medium"] = {{ 95, 97, 99 }};
    brake.sets["Full"] = {{ 98, 100, 100, 100 }};

    vector<Rule> ruleBase = { // Column, Row, Throttle, Brake
        {"Stopped", "At", "No", "Full"},
        {"Stopped", "Very Near", "Very Slow", "Full"},
        {"Very Slow", "At", "No", "Full"},
        {"Very Slow", "Very Near", "Very Slow", "Medium"},
        {"Very Slow", "Near", "Very Slow", "Slow"},
        {"Slow", "At", "No", "Full"},
        {"Slow", "Very Near", "Very Slow", "Medium"},
        {"Slow", "Near", "Slow", "Very Slow"},
        {"Medium Fast", "Medium Far", "Medium", "Very Slow"},
        {"Medium Fast", "Far", "Full", "No"},
        {"Fast", "Medium Far", "Medium", "Very Slow"},
        {"Fast", "Far", "Full", "No"}
    };

    double inputSpeed = 3.0;
    double inputDistance = 1.8;

    cout << "--- Mamdani Fuzzy Logic Controller: Train Control ---" << endl;
    cout << "Crisp Inputs:" << endl;
    cout << "  - Speed: " << inputSpeed << " km/hr" << endl;
    cout << "  - Distance: " << inputDistance << " m" << endl;
    cout << endl;

    auto speedMemberships = fuzzify(inputSpeed, speed);
    auto distanceMemberships = fuzzify(inputDistance, distance);

    cout << "--- Step 1: Fuzzification Results ---" << endl;
    cout << "Speed Memberships (μ):" << endl;
    for(const auto& pair : speedMemberships) cout << "  - " << pair.first << ": " << pair.second << endl;
    cout << "Distance Memberships (μ):" << endl;
    for(const auto& pair : distanceMemberships) cout << "  - " << pair.first << ": " << pair.second << endl;
    cout << endl;

    map<string, double> aggregatedThrottle;
    map<string, double> aggregatedBrake;

    cout << "--- Step 2: Rule Evaluation (Fired Rules) ---" << endl;
    for (const auto& rule : ruleBase) {
        if (speedMemberships.count(rule.speed_is) && distanceMemberships.count(rule.distance_is)) {
            double ruleStrength = min(speedMemberships.at(rule.speed_is), distanceMemberships.at(rule.distance_is));
            
            cout << "IF Speed is " << rule.speed_is << " AND Distance is " << rule.distance_is<<" then ("
                      << rule.brake_is<<" Brake, "<<rule.throttle_is<<" Throttle) => Strength: " << ruleStrength << endl;

            if (aggregatedThrottle.find(rule.throttle_is) == aggregatedThrottle.end() || ruleStrength > aggregatedThrottle[rule.throttle_is]) {
                aggregatedThrottle[rule.throttle_is] = ruleStrength;
            }
            if (aggregatedBrake.find(rule.brake_is) == aggregatedBrake.end() || ruleStrength > aggregatedBrake[rule.brake_is]) {
                aggregatedBrake[rule.brake_is] = ruleStrength;
            }
        }
    }
    cout << endl;

    double crispThrottle = defuzzify(aggregatedThrottle, throttle);
    double crispBrake = defuzzify(aggregatedBrake, brake);

    cout << "--- Step 3: Defuzzification (Centroid Method) ---" << endl;
    cout << "Aggregated Throttle Outputs:" << endl;
    for(const auto& pair : aggregatedThrottle) cout << "  - " << pair.first << " (Clipped at " << pair.second << ")" << endl;
    cout << "Aggregated Brake Outputs:" << endl;
    for(const auto& pair : aggregatedBrake) cout << "  - " << pair.first << " (Clipped at " << pair.second << ")" << endl;
    cout << endl;

    cout << "--- FINAL CRISP OUTPUTS ---" << endl;
    cout << "Recommended Throttle Setting: " << crispThrottle << " %" << endl;
    cout << "Recommended Brake Setting: " << crispBrake << " %" << endl;

    return 0;
}