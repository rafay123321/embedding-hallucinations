from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv 
from pathlib import Path

load_dotenv(override=True)

output_dir = Path.cwd() / "outputs"
output_dir.mkdir(exist_ok=True)
# Load the model
model = SentenceTransformer('msmarco-distilbert-base-tas-b')

def compare_sentences(sentence1, sentence2, model, category, subcategory):
    """Compare two sentences and return their cosine similarity"""
    embeddings = model.encode([sentence1, sentence2])
    cosine_score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return {
        'Category': category,
        'Subcategory': subcategory,
        'Sentence1': sentence1,
        'Sentence2': sentence2,
        'Similarity': cosine_score
    }

# List to store all results
results = []

# 1. Capitalization
base_sentence = "Ritesh visited New York last summer and enjoyed the museums."

capitalization_tests = [
    ("All lowercase", base_sentence.lower()),
    ("All uppercase", base_sentence.upper()),
    ("Title case", base_sentence.title()),
    ("Random case", "Ritesh vIsiTed nEw yOrK lAsT sUmMeR aNd eNjOyEd tHe mUsEuMs."),
    ("Proper noun lowercase", "Ritesh visited new york last summer and enjoyed the museums."),
    ("Proper noun lowercase", "Ritesh visited New York last summer and enjoyed the museums."),
]

for test_name, test_sentence in capitalization_tests:
    results.append(compare_sentences(base_sentence, test_sentence, model, "Capitalization", test_name))

# 2. Whitespace variations
whitespace_tests = [
    ("Extra spaces", "Ritesh  visited  New  York  last  summer  and  enjoyed  the  museums."),
    ("Leading spaces", "   Ritesh visited New York last summer and enjoyed the museums."),
    ("Trailing spaces", "Ritesh visited New York last summer and enjoyed the museums.   "),
    ("No spaces", "RiteshvisitedNewYorklastsummerandenjovedthemuseums."),
    ("Newlines", "Ritesh visited New York\nlast summer and\nenjoyed the museums."),
    ("Tabs", "Ritesh visited\tNew York\tlast summer\tand enjoyed\tthe museums."),
]

for test_name, test_sentence in whitespace_tests:
    results.append(compare_sentences(base_sentence, test_sentence, model, "Whitespace", test_name))

# 3. Negation
negation_base = "The movie was good and I enjoyed it."
negation_tests = [
    ("Simple negation", "The movie was not good and I enjoyed it."),
    ("Contraction negation", "The movie wasn't good and I enjoyed it."),
    ("Double negation", "The movie wasn't bad and I enjoyed it."),
    ("Multiple negations", "The movie was not good and I didn't enjoy it."),
    ("Partial negation", "The movie was good but I didn't enjoy it."),
]

for test_name, test_sentence in negation_tests:
    results.append(compare_sentences(negation_base, test_sentence, model, "Negation", test_name))

# 4. Special characters
special_chars_tests = [
    ("Punctuation removed", "Ritesh visited New York last summer and enjoyed the museums"),
    ("Extra punctuation", "Ritesh visited New York, last summer, and enjoyed the museums!!!"),
    ("Symbols added", "Ritesh visited New York (last summer) and enjoyed the museums #vacation"),
    ("Emojis", "Ritesh visited New York last summer and enjoyed the museums 🗽🏙️"),
    ("ASCII art", "Ritesh visited New York last summer and enjoyed the museums ¯\\_(ツ)_/¯"),
]

for test_name, test_sentence in special_chars_tests:
    results.append(compare_sentences(base_sentence, test_sentence, model, "Special Characters", test_name))

# 5. Word order
word_order_tests = [
    ("Reversed sentence", "The museums enjoyed and summer last York New visited Ritesh."),
    ("Shuffled words", "York visited summer Ritesh enjoyed last museums the New and."),
    ("Slight reordering", "Last summer, Ritesh visited New York and enjoyed the museums."),
    ("Active to passive", "The museums in New York were enjoyed by Ritesh last summer."),
    ("Clauses reordered", "Ritesh enjoyed the museums and visited New York last summer."),
]

for test_name, test_sentence in word_order_tests:
    results.append(compare_sentences(base_sentence, test_sentence, model, "Word Order", test_name))

# 6. Synonyms and paraphrasing
synonyms_tests = [
    ("Synonyms", "Ritesh traveled to New York last summer and liked the galleries."),
    ("Light paraphrase", "Ritesh went to NYC last summer and had fun at the museums."),
    ("Moderate paraphrase", "Last year during the summer months, Ritesh took a trip to New York where he found the museums enjoyable."),
    ("Heavy paraphrase", "During his vacation to the Big Apple in the previous year's warm season, Ritesh expressed appreciation for the cultural institutions he visited."),
    ("Same meaning, different words", "Ritesh had a good time at the exhibits when he was in New York during the previous summer."),
]

for test_name, test_sentence in synonyms_tests:
    results.append(compare_sentences(base_sentence, test_sentence, model, "Synonyms & Paraphrasing", test_name))

# 7. Spelling errors and typos
spelling_tests = [
    ("Minor typo", "Ritesh viseted New York last summer and enjoyed the museums."),
    ("Multiple typos", "Jhon visted Nwe Yrok last summre and enoyded the meusums."),
    ("Phonetic spelling", "Jon vizited New Yirk last sumer and enjoyd the muzeums."),
    ("Character swaps", "Ritesh vistied New Yrok alst summer and enjyoed the mueseums."),
    ("Missing letters", "Jhn vstd Nw Yrk lst smmr and enjyd the msms."),
]

for test_name, test_sentence in spelling_tests:
    results.append(compare_sentences(base_sentence, test_sentence, model, "Spelling & Typos", test_name))

# 8. Named entity variations
entity_tests = [
    ("Person name change", "Mike visited New York last summer and enjoyed the museums."),
    ("Location change", "Ritesh visited Chicago last summer and enjoyed the museums."),
    ("Time change", "Ritesh visited New York last winter and enjoyed the museums."),
    ("Entity abbreviation", "Ritesh visited NYC last summer and enjoyed the museums."),
    ("Entity expansion", "Ritesh visited New York City, United States of America last summer and enjoyed the museums."),
]

for test_name, test_sentence in entity_tests:
    results.append(compare_sentences(base_sentence, test_sentence, model, "Named Entities", test_name))

# 9. Grammatical variations
grammar_tests = [
    ("Tense change", "Ritesh visits New York last summer and enjoys the museums."),
    ("Plurality", "Ritesh visited New York last summer and enjoyed a museum."),
    ("Voice change", "New York was visited by Ritesh last summer and the museums were enjoyed."),
    ("Article change", "Ritesh visited a New York last summer and enjoyed some museums."),
    ("Pronoun substitution", "He visited New York last summer and enjoyed the museums."),
]

for test_name, test_sentence in grammar_tests:
    results.append(compare_sentences(base_sentence, test_sentence, model, "Grammar", test_name))

# 10. Filler words and verbosity
filler_tests = [
    ("Added fillers", "Well, Ritesh basically just visited New York last summer and, you know, enjoyed the museums and stuff."),
    ("Very verbose", "If I may say so, our friend Ritesh, whom we all know well, took an exciting journey to the magnificent city of New York during the pleasant summer months of last year, where he subsequently proceeded to experience considerable enjoyment while exploring the numerous fascinating museums available in that location."),
    ("Very concise", "Ritesh: NYC trip, summer. Enjoyed museums."),
    ("Repetition", "Ritesh visited, visited New York last summer, last summer and enjoyed, really enjoyed the museums."),
    ("Hedging language", "Ritesh possibly visited New York last summer and perhaps enjoyed the museums to some extent."),
]

for test_name, test_sentence in filler_tests:
    results.append(compare_sentences(base_sentence, test_sentence, model, "Verbosity", test_name))

# 11. Contractions and expansions
contraction_base = "I don't think they'll be able to attend the party."
contraction_tests = [
    ("Expanded contractions", "I do not think they will be able to attend the party."),
    ("Mixed contractions", "I don't think they will be able to attend the party."),
    ("Additional contractions", "I don't think they'll've been able t'attend the party."),
    ("Formal expansion", "I do not believe they will be capable of attending the gathering."),
    ("Informal contractions", "I dunno if they gonna make it to the party."),
]

for test_name, test_sentence in contraction_tests:
    results.append(compare_sentences(contraction_base, test_sentence, model, "Contractions", test_name))

# 12. Missing information
missing_info_tests = [
    ("Subject omitted", "Visited New York last summer and enjoyed the museums."),
    ("Object omitted", "Ritesh visited New York last summer and enjoyed."),
    ("Truncated sentence", "Ritesh visited New York last..."),
    ("Key detail omitted", "Ritesh visited last summer and enjoyed the museums."),
    ("Partial information", "Ritesh New York museums summer."),
]

for test_name, test_sentence in missing_info_tests:
    results.append(compare_sentences(base_sentence, test_sentence, model, "Missing Information", test_name))

# 13. Language mixing and code-switching
language_tests = [
    ("Light Spanish", "Ritesh visitó New York last summer y enjoyed the museums."),
    ("Light French", "Ritesh a visité New York last summer et a apprécié les museums."),
    ("Spanglish", "Ritesh visited Nueva York el verano pasado and enjoyed los museos."),
    ("Transliteration", "Dzhon vizited N'yu York proshlogo leta i naslazhdalsya muzeyami."),
    ("Machine translation", "Ritesh ha visitato New York la scorsa estate e si è divertito nei musei."),
]

for test_name, test_sentence in language_tests:
    results.append(compare_sentences(base_sentence, test_sentence, model, "Language Mixing", test_name))



base_sentence = "She completed her degree before starting the job"

# temporal_direction
temporal_direction = [
    ("temporal_direction", "She started her job before completing her degree"),
]

for test_name, test_sentence in temporal_direction:
    results.append(compare_sentences(base_sentence, test_sentence, model, "temporal_direction", test_name))
###

base_sentence = "The company barely exceeded earnings expectations"

# quant_threshold
quant_threshold = [
    ("quant_threshold", "The company significantly missed earnings expectations"),
]

for test_name, test_sentence in quant_threshold:
    results.append(compare_sentences(base_sentence, test_sentence, model, "quant_threshold", test_name))
###

base_sentence = "If the treatment works, symptoms should improve"

# hypo_fact
hypo_fact = [
    ("hypo_fact", "The treatment works and symptoms have improved"),
]

for test_name, test_sentence in hypo_fact:
    results.append(compare_sentences(base_sentence, test_sentence, model, "hypo_fact", test_name))
###


base_sentence = "The meeting ran significantly shorter than planned"

# hypo_fact
scaler_inversion = [
    ("scaler_inversion", "The meeting ran significantly longer than planned"),
]

for test_name, test_sentence in scaler_inversion:
    results.append(compare_sentences(base_sentence, test_sentence, model, "scaler_inversion", test_name))
###

base_sentence = "The patient presents with tachycardia"

# medicine_domain_based
medicine_domain_based = [
    ("medicine_domain_based", "The patient presents with bradycardia"),
]

for test_name, test_sentence in medicine_domain_based:
    results.append(compare_sentences(base_sentence, test_sentence, model, "medicine_domain_based", test_name))
###

base_sentence = "Plaintiff bears the burden of proof"

# legal_domain_based
legal_domain_based = [
    ("legal_domain_based", "Defendant bears the burden of proof"),
]

for test_name, test_sentence in legal_domain_based:
    results.append(compare_sentences(base_sentence, test_sentence, model, "legal_domain_based", test_name))
###

base_sentence = "Research supports the efficacy of this treatment"

# attribution
attribution = [
    ("attribution", "Research questions the efficacy of this treatment"),
]

for test_name, test_sentence in attribution:
    results.append(compare_sentences(base_sentence, test_sentence, model, "attribution", test_name))
###

base_sentence = "The procedure takes about 5 minutes"

# 6. unit of time
numerical = [
    ("Numerical", "The procedure takes about 5 hours"),
]

for test_name, test_sentence in numerical:
    results.append(compare_sentences(base_sentence, test_sentence, model, "Numerical", test_name))
###

base_sentence = "The tumor is 2 centimeters in diameter"

# unit_conversion
unit_conversion = [
    ("unit_conversion", "The tumor is 2 inches in diameter"),
]

for test_name, test_sentence in unit_conversion:
    results.append(compare_sentences(base_sentence, test_sentence, model, "unit_conversion", test_name))
###


base_sentence = "Maintain speeds under 30 mph"

# speed and miles
miles = [
    ("miles", "Maintain speeds under 30 kph"),
]

for test_name, test_sentence in miles:
    results.append(compare_sentences(base_sentence, test_sentence, model, "miles", test_name))
###

base_sentence = "The product costs between $50-$100"

# exact vs range
exact = [
    ("exact", "The product costs exactly $101"),
]

for test_name, test_sentence in exact:
    results.append(compare_sentences(base_sentence, test_sentence, model, "exact", test_name))
###


base_sentence = "The patient's fever was 101°F"

# domain_significance
domain_significance = [
    ("domain_significance", "The patient's fever was 104°F"),
]

for test_name, test_sentence in domain_significance:
    results.append(compare_sentences(base_sentence, test_sentence, model, "domain_significance", test_name))
###


base_sentence = "Only 5% of patients experienced side effects"

# Percentages
Percentages = [
    ("Percentages", "Up to 95% of patients experienced side effects"),
]

for test_name, test_sentence in Percentages:
    results.append(compare_sentences(base_sentence, test_sentence, model, "Percentages", test_name))
###


base_sentence = "Submit your application by 12/10/2023"

# Date and time
date_time = [
    ("date_time", "Submit your application by 10/12/2023"),
]

for test_name, test_sentence in date_time:
    results.append(compare_sentences(base_sentence, test_sentence, model, "date_time", test_name))
###

base_sentence = "The results showed a significant difference (p<0.05)"

# statistics
stats = [
    ("stats", "The results showed significant difference (p>0.05)"),
]

for test_name, test_sentence in stats:
    results.append(compare_sentences(base_sentence, test_sentence, model, "stats", test_name))
###

base_sentence = "If demand increases, prices will rise"

# Counterfactual
Counterfactual = [
    ("Counterfactual", "If demand decreases, prices will fall"),
]

for test_name, test_sentence in Counterfactual:
    results.append(compare_sentences(base_sentence, test_sentence, model, "Counterfactual", test_name))
###

base_sentence = "This component is a type of resistor"

# Taxonomic
Taxonomic = [
    ("Taxonomic", "This resistor is a type of component"),
]

for test_name, test_sentence in Taxonomic:
    results.append(compare_sentences(base_sentence, test_sentence, model, "Taxonomic", test_name))
###

base_sentence = "Add the eggs after heating the oil"

# Procedural
Procedural = [
    ("Procedural", "Add the eggs before heating the oil"),
]

for test_name, test_sentence in Procedural:
    results.append(compare_sentences(base_sentence, test_sentence, model, "Procedural", test_name))
###


base_sentence = "This material is aluminum"

# Comparison
Comparison = [
    ("Comparison", "This material resembles aluminum"),
]

for test_name, test_sentence in Comparison:
    results.append(compare_sentences(base_sentence, test_sentence, model, "Comparison", test_name))
###


base_sentence = "The market is climbing a wall of worry"

# Metaphorical
Metaphorical = [
    ("Metaphorical", "Rock climbers are scaling a worrying wall"),
]

for test_name, test_sentence in Metaphorical:
    results.append(compare_sentences(base_sentence, test_sentence, model, "Metaphorical", test_name))
###


base_sentence = "What caused the system to fail?"

# Presupposition
Presupposition = [
    ("Presupposition", "Did the system fail?"),
]

for test_name, test_sentence in Presupposition:
    results.append(compare_sentences(base_sentence, test_sentence, model, "Presupposition", test_name))
###

base_sentence = "The car is to the left of the tree"

# References
Reference = [
    ("Reference", "the tree is to the right of the car"),
]

for test_name, test_sentence in Reference:
    results.append(compare_sentences(base_sentence, test_sentence, model, "Reference", test_name))
###

base_sentence = "The Morning Star is visible at dawn"

# Extensional
Extensional = [
    ("Extensional", "The Evening Star is visible at dawn"),
]

for test_name, test_sentence in Extensional:
    results.append(compare_sentences(base_sentence, test_sentence, model, "Extensional", test_name))
###

# Convert results to DataFrame
df = pd.DataFrame(results)

# Visualization
plt.figure(figsize=(15, 10))
sns.barplot(data=df, x='Similarity', y='Subcategory', hue='Category', dodge=False)
plt.title('Embedding Similarity Across Different Text Variations')
plt.xlabel('Cosine Similarity')
plt.ylabel('Variation Type')
plt.tight_layout()
plt.savefig(f"{output_dir}/embedding_comparisons.png")

# Print summary statistics by category
print(df.groupby('Category')['Similarity'].describe())

# Save results to CSV
df.to_csv(f"{output_dir}/embedding_comparison_results.csv", index=False)

print("Total comparisons:", len(df))
print("Average similarity across all tests:", df['Similarity'].mean())
print("Min similarity:", df['Similarity'].min(), "in category:", df.loc[df['Similarity'].idxmin(), 'Category'], "-", df.loc[df['Similarity'].idxmin(), 'Subcategory'])
print("Max similarity:", df['Similarity'].max(), "in category:", df.loc[df['Similarity'].idxmax(), 'Category'], "-", df.loc[df['Similarity'].idxmax(), 'Subcategory'])

# Optional: Show most and least affected categories
category_impact = df.groupby('Category')['Similarity'].mean().sort_values()
print("\nCategories from most to least impact on embeddings:")
for category, avg_similarity in category_impact.items():
    print(f"{category}: {avg_similarity:.4f}")