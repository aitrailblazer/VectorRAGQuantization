import random

# Define lists of subjects, verbs, domains, and adverbs.
subjects = [
    "AI in the legal field",
    "AI in agriculture",
    "Self-driving cars",
    "Neural networks",
    "Natural language processing",
    "AI in cybersecurity",
    "Ethical AI",
    "AI-powered fraud detection",
    "Recommendation engines",
    "Reinforcement learning",
    "AI in marketing",
    "AI in logistics",
    "Quantum computing",
    "Edge AI",
    "Voice recognition",
    "Big data analytics",
    "Machine learning",
    "Robotics",
    "Generative AI",
    "Explainable AI"
]

verbs = [
    "reduces costs in",
    "improves",
    "integrates into",
    "simplifies",
    "optimizes",
    "streamlines",
    "advances",
    "revolutionizes",
    "reshapes",
    "transforms",
    "creates opportunities for",
    "ensures",
    "predicts",
    "aids in"
]

domains = [
    "translation accuracy",
    "crop yields",
    "artificial intelligence research",
    "protein structure decoding",
    "education personalization",
    "financial risk management",
    "manufacturing processes",
    "cybersecurity",
    "supply chain optimization",
    "human-computer interaction",
    "document review",
    "traffic management",
    "drug discovery",
    "astronomical data analysis",
    "real estate valuation",
    "energy efficiency",
    "customer experience",
    "e-commerce engagement",
    "climate science predictions",
    "smart city planning"
]

adverbs = [
    "dramatically",
    "significantly",
    "remarkably",
    "rapidly",
    "consistently",
    "substantially",
    "notably",
    "markedly"
]

# Define a set of sentence templates to mix the components.
templates = [
    "{subject} {verb} {domain}.",
    "{subject} is transforming {domain}.",
    "{subject} is revolutionizing {domain}.",
    "{subject} {adv} {verb} {domain}.",
    "{subject} is {adv} transforming {domain}.",
    "{subject} is {adv} revolutionizing {domain}.",
    "{adv}, {subject} {verb} {domain}.",
    "According to recent research, {subject} {verb} {domain}.",
    "Recent studies show that {subject} is {adv} transforming {domain}."
]

def generate_example():
    subject = random.choice(subjects)
    verb = random.choice(verbs)
    domain = random.choice(domains)
    adv = random.choice(adverbs)
    template = random.choice(templates)
    # Only fill in {adv} if the template expects it.
    if "{adv}" in template:
        example = template.format(subject=subject, verb=verb, domain=domain, adv=adv)
    else:
        example = template.format(subject=subject, verb=verb, domain=domain)
    return example

# Generate 10,000 examples.
generated_examples = [generate_example() for _ in range(10000)]

# Print the examples to the console.
for example in generated_examples:
    print(example)

# Save the examples to a text file.
with open("generated_examples.txt", "w", encoding="utf-8") as f:
    for example in generated_examples:
        f.write(example + "\n")