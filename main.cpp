//
//  main.cpp
//  project5-ml
//
//  Created by Abrar Quazi on 6/11/17.
//  Copyright © 2017 Abrar Quazi. All rights reserved.
//

#include <iostream>
#include <cmath>
#include <iomanip>
#include <string>
#include "csvstream.h"
#include <set>
#include <map>
using namespace std;


int main(int argc, const char * argv[]) {
    string debug = "off" ;
    cout << setprecision(3);
    if (argc != 3 && argc!= 4) {
       cout << "Usage: main TRAIN_FILE TEST_FILE [--debug]" << endl;
       return -1;
    }
    else if (argc == 4) {
        debug = "on";
        if (!strcmp((argv[3])," --debug ")) {
            cout << "Usage: main TRAIN_FILE TEST_FILE [--debug]" << endl;
            return -1;
        }
    }
    
    try {
    csvstream trainfile (argv[1]);
    csvstream testfile (argv[2]);

    map<string,string> rowTrain;
    map<string,string> rowTest;
    map<string,double> vocabulary;
    map<string,double>tag;
    
     map<string,map<string,double>> log_and_label;
    
    //used to check number of posts with label C that contain W
    //outer key will be label, inner key will be the Word
    map<string,map<string,double>> wordsInTag;
    set<string> content;

    double numPosts = 0;
    while(trainfile>>rowTrain) {
        numPosts++;
        //retrieves the content of post and parses it into indiviual words
        string columnContent;
        columnContent = rowTrain["content"];
        istringstream is(columnContent);

        //checks number of posts with that label
        string label = rowTrain["tag"];

         tag[label] += 1;

        if (debug == "on") {
            if (numPosts ==  1) {
                cout << "training data:" << endl;
            }
            cout <<"  label = " << label <<", content = " << columnContent << endl;
        }
    

        // makes a set where content of post has no duplicates
        string word;
        while (is >> word ) {
            content.insert(word);
        }
        //inserts words for post if not seen before, otherwise just increments the number of instances of the word. Also keeps in check the number of posts with label C that contain word w.
        for (auto i: content) {
            vocabulary[i] += 1;
            wordsInTag[label][i] += 1;
          //  cout << label << " " << i << " " << wordsInTag[label][i] << endl;
        }

       content.clear();
    }
    cout << "trained on " << numPosts <<" examples" << endl;
    if (debug == "on") {
            cout <<"vocabulary size = " << vocabulary.size() << endl;
       
        
          }
    cout << endl;

   
    
    // Teaching it
    
        if (debug == "on") {
                cout << "classes:" << endl;
        }
        for(auto key: wordsInTag ) {
            double log_likelihood = 0;
            string label = key.first;

            for (auto i: wordsInTag[label]) {
                //If word exists inside label c, will calculate
                if (wordsInTag[label].find(i.first) != wordsInTag[label].end())   {
                    log_likelihood = log((wordsInTag[label][i.first])/tag[label]);

                }
                else {
                    if (vocabulary.find(i.first) == vocabulary.end()) {
                        log_likelihood = log(1/numPosts);

                    }
                    else {
                        log_likelihood = log(vocabulary[i.first]/numPosts);

                    }
             
                   }
                log_and_label[label][i.first] = log_likelihood;
                
            }
       
            
        }
        if (debug == "on") {
            for (auto key: wordsInTag) {
                string label = key.first;
                double log_prior_learning = log((tag[label])/numPosts);
                cout << "  " << label << ", " << tag[label] << " examples, log-prior = " << log_prior_learning << endl;
                }
            cout << "classifier parameters:" << endl;
            for (auto key :log_and_label) {
                string label = key.first;
                for(auto i: log_and_label[label]) {
                    cout << "  " << label << ":" << i.first << ", count = " << wordsInTag[label][i.first] <<  ", log-likelihood = " << log_and_label[label][i.first] << endl;
                }
            
            }
            cout << endl;
        }

    
    int numCorrect = 0;
    int numTests = 0;
    while(testfile >> rowTest) {
       map<string, double> log_scores;
       numTests++;
        string columnContent;
        columnContent = rowTest["content"];
        string correctLabel = rowTest["tag"];
        istringstream is(columnContent);
        string word;
        
        while(is >> word) {
            content.insert(word);
        }
        for (auto key :log_and_label) {
            string label = key.first;
             double log_probability_score = 0;
             double log_prior = 0;
            //cout << label << endl;
            double log_likelihood = 0;
            for(auto i: content) {
                if(log_and_label[label].find(i) != log_and_label[label].end()) {
                    log_probability_score += log_and_label[label][i];
                   // cout << i << endl;
                }
                else {
                    if (vocabulary.find(i) == vocabulary.end()) {
                        log_likelihood = log(1/numPosts);
                        log_probability_score += log_likelihood;
                    //cout << i << endl;
                    }
                    else {
                        log_likelihood = log(vocabulary[i]/numPosts);
                        log_probability_score += log_likelihood;
                    //cout << i << endl;
                    }
             
                   }
            }
            log_prior = log((tag[label])/numPosts);
            log_scores[label] = log_probability_score + log_prior;
        }
        string predictedLabel = log_scores.begin()->first;
        double max = log_scores.begin()->second;
        for (auto i = log_scores.begin(); i != log_scores.end(); ++i) {
            
            if (i->second > max) {
                max = i->second;
                predictedLabel = i->first;
            }
        }
        
       if (numTests == 1) {
            cout << "test data:" << endl;
       }
       cout << "  correct = " << correctLabel << ", predicted = " << predictedLabel << ", log-probability score = " << max << endl;
        cout << "  content = " << columnContent <<endl << endl;
        if (correctLabel == predictedLabel) {
           numCorrect++;
        }
        content.clear();
    
    }
        cout << "performance: " << numCorrect << " / " << numTests << " posts predicted correctly" <<endl;

    }
    catch( csvstream_exception msg) {
        
        cout << msg.msg << endl;
    }
    
    return 0;
}
