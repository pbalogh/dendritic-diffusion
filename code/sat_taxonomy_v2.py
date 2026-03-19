"""
Supersaturation Taxonomy v2: 200 seeds, lower threshold (0.10)
==============================================================

Changes from v1:
- 200 seeds (40 per discourse type) instead of 50
- Continuation threshold lowered from 0.20 to 0.10
- Seeds drawn from diverse sources for ecological validity
- Longer max_branches (12) to capture full decay curves
- Records per-token resolution order within first branch

Goal: enough multi-branch traces to validate structural/semantic ratio
finding and produce preliminary shape distributions for §5.5.
"""
import os, sys
os.environ['HF_HOME'] = '/data/hf_cache'
sys.path.insert(0, '/data/dllm')

import torch
import torch.nn.functional as F
import re, json, time, math
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MASK_ID = 126336
model_name = 'GSAI-ML/LLaDA-8B-Base'

print(f"Loading {model_name}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='/data/hf_cache',
                                           trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, cache_dir='/data/hf_cache', quantization_config=bnb_config,
    device_map='auto', trust_remote_code=True,
).eval()
device = next(p.device for p in model.parameters() if p.device.type == 'cuda')

END_IDS = set()
for tok_str in ['<|eot_id|>', '<|end_of_text|>']:
    tid = tokenizer.convert_tokens_to_ids(tok_str)
    if tid is not None and tid != tokenizer.unk_token_id:
        END_IDS.add(tid)
if tokenizer.eos_token_id:
    END_IDS.add(tokenizer.eos_token_id)

STOPS = {'the','a','an','is','are','was','were','be','been','being',
         'have','has','had','do','does','did','will','would','could',
         'should','may','might','can','shall','must',
         'of','in','to','for','with','on','at','by','from','as',
         'into','through','during','before','after','between',
         'and','or','but','nor','yet','so','because','although',
         'while','when','where','if','then','than','that','which',
         'who','whom','whose','what','how','whether',
         'it','its','this','that','these','those','he','she','they',
         'his','her','their','him','them','we','our','you','your',
         'not','no','also','very','more','most','much','many'}

print(f"Model loaded.")


def extract_sentences(tokens):
    clean = []
    for t in tokens:
        if t in END_IDS:
            break
        clean.append(t)
    text = tokenizer.decode(clean).strip()
    sent_ends = [m.end() for m in re.finditer(r'[.!?]+(?:\s|$)', text)]
    if sent_ends:
        text = text[:sent_ends[-1]].strip()
    elif len(text) > 30:
        text = text.strip().rstrip(',;:') + '.'
    if not text or len(text) < 5:
        return [], ''
    return tokenizer.encode(text, add_special_tokens=False), text


def internal_repetition(text, n=3):
    words = text.lower().split()
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / max(1, len(grams))


def denoise_branch(input_ids, start, end, steps=28, temperature=1.15):
    for step in range(steps):
        t = torch.tensor([input_ids], device=device)
        with torch.no_grad():
            logits = model(t).logits[0].float()

        mask_positions = [i for i in range(start, end) if input_ids[i] == MASK_ID]
        if not mask_positions:
            break

        cands = []
        for i in mask_positions:
            scaled = logits[i] / temperature
            probs = F.softmax(scaled, dim=-1)
            topk_probs, topk_ids = probs.topk(10)
            topk_probs = topk_probs / topk_probs.sum()
            chosen_idx = torch.multinomial(topk_probs, 1).item()
            idx = topk_ids[chosen_idx].item()
            p = probs[idx].item()
            cands.append((i, idx, p))

        cands.sort(key=lambda x: -x[2])
        n = max(1, len(cands) // max(1, steps - step))
        for pos, tok, _ in cands[:n]:
            input_ids[pos] = tok

    return input_ids


def measure_supersaturation(text_ids, position, probe_tokens=24):
    probed = list(text_ids[:position]) + [MASK_ID] * probe_tokens + list(text_ids[position:])
    if len(probed) > 2048:
        return None

    t = torch.tensor([probed], device=device)
    with torch.no_grad():
        logits = model(t).logits[0].float()

    confs = []
    struct_confs = []
    semantic_confs = []
    top_preds = []

    for i in range(position, position + probe_tokens):
        probs = F.softmax(logits[i], dim=-1)
        top_prob, top_id = probs.max(dim=-1)

        if top_id.item() in END_IDS:
            continue

        conf = top_prob.item()
        confs.append(conf)

        tok_text = tokenizer.decode([top_id.item()]).strip().lower()
        if tok_text in STOPS or len(tok_text) <= 1:
            struct_confs.append(conf)
        else:
            semantic_confs.append(conf)

        if len(top_preds) < 6:
            top_preds.append(tokenizer.decode([top_id.item()]).strip())

    if not confs:
        return {'total': 0.0, 'structural': 0.0, 'semantic': 0.0,
                'n_struct': 0, 'n_semantic': 0, 'top_preds': []}

    return {
        'total': sum(confs) / len(confs),
        'structural': sum(struct_confs) / max(1, len(struct_confs)) if struct_confs else 0.0,
        'semantic': sum(semantic_confs) / max(1, len(semantic_confs)) if semantic_confs else 0.0,
        'n_struct': len(struct_confs),
        'n_semantic': len(semantic_confs),
        'top_preds': top_preds,
    }


# Lower threshold: 0.10 instead of 0.20
SAT_THRESHOLD = 0.10

def grow_with_trace(seed_text, max_branches=12, max_tokens=512):
    prompt_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    text_ids = list(prompt_ids)
    plen = len(prompt_ids)
    all_text = ""
    trace = []

    for bi in range(max_branches):
        if len(text_ids) - plen >= max_tokens:
            break

        bstart = len(text_ids)
        n = min(64, max_tokens - (len(text_ids) - plen))
        if n < 8:
            break

        ids = text_ids + [MASK_ID] * n
        ids = denoise_branch(ids, bstart, len(ids), 28, 1.15)

        raw = ids[bstart:]
        clean_tokens, clean_text = extract_sentences(raw)

        if not clean_text or len(clean_tokens) < 3:
            sat = measure_supersaturation(text_ids, len(text_ids))
            if sat:
                trace.append({
                    'branch': bi, 'sat_total': round(sat['total'], 4),
                    'sat_struct': round(sat['structural'], 4),
                    'sat_sem': round(sat['semantic'], 4),
                    'n_struct': sat['n_struct'], 'n_semantic': sat['n_semantic'],
                    'tokens_added': 0, 'text': '', 'status': 'empty',
                })
            break

        if internal_repetition(clean_text) > 0.35:
            sat = measure_supersaturation(text_ids, len(text_ids))
            if sat:
                trace.append({
                    'branch': bi, 'sat_total': round(sat['total'], 4),
                    'sat_struct': round(sat['structural'], 4),
                    'sat_sem': round(sat['semantic'], 4),
                    'n_struct': sat['n_struct'], 'n_semantic': sat['n_semantic'],
                    'tokens_added': len(clean_tokens), 'text': clean_text[:100],
                    'status': 'repetitive',
                })
            break

        text_ids.extend(clean_tokens)
        all_text += " " + clean_text

        sat = measure_supersaturation(text_ids, len(text_ids))
        if sat is None:
            trace.append({
                'branch': bi, 'sat_total': 0.0, 'sat_struct': 0.0, 'sat_sem': 0.0,
                'tokens_added': len(clean_tokens), 'text': clean_text[:100],
                'status': 'overflow',
            })
            break

        trace.append({
            'branch': bi, 'sat_total': round(sat['total'], 4),
            'sat_struct': round(sat['structural'], 4),
            'sat_sem': round(sat['semantic'], 4),
            'n_struct': sat.get('n_struct', 0), 'n_semantic': sat.get('n_semantic', 0),
            'top_preds': sat.get('top_preds', []),
            'tokens_added': len(clean_tokens), 'text': clean_text[:100],
            'status': 'ok',
        })

        if sat['total'] < SAT_THRESHOLD:
            break

    final_text = tokenizer.decode(text_ids[plen:]).strip()
    return {
        'trace': trace,
        'final_text': final_text,
        'total_tokens': len(text_ids) - plen,
        'n_branches': len([t for t in trace if t['status'] == 'ok']),
    }


# ============================================================
# 200 SEEDS — 40 per discourse type
# ============================================================

SEEDS = {
    'expository': [
        ("Immune system", "The human immune system is a complex network of cells, tissues, and organs that work together to defend the body against harmful invaders."),
        ("Photosynthesis", "Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose molecules."),
        ("DNA replication", "DNA replication is the biological process by which a cell copies its DNA before cell division."),
        ("Plate tectonics", "Plate tectonics is the scientific theory that Earth's outer shell is divided into several plates that glide over the mantle."),
        ("Neural networks", "Artificial neural networks are computing systems inspired by biological neural networks that constitute animal brains."),
        ("Water cycle", "The water cycle describes the continuous movement of water on, above, and below the surface of the Earth."),
        ("Black holes", "A black hole is a region of spacetime where gravity is so strong that nothing, not even light, has enough energy to escape."),
        ("Quantum mechanics", "Quantum mechanics is a fundamental theory in physics that describes the behavior of matter and energy at the atomic and subatomic scale."),
        ("Evolution", "Evolution by natural selection is the process by which organisms with favorable traits are more likely to reproduce and pass on their genes."),
        ("Digestive system", "The human digestive system is a complex series of organs and glands that processes the food we eat."),
        ("Mitochondria", "Mitochondria are membrane-bound organelles found in the cytoplasm of eukaryotic cells that generate most of the cell's supply of adenosine triphosphate."),
        ("Ocean currents", "Ocean currents are continuous, directed movements of seawater generated by forces acting upon the water, including wind, temperature, salinity, and the Coriolis effect."),
        ("Periodic table", "The periodic table is a tabular arrangement of chemical elements, organized by their atomic number, electron configuration, and recurring chemical properties."),
        ("Nervous system", "The nervous system is a highly complex part of an animal that coordinates its actions and sensory information by transmitting signals to and from different parts of its body."),
        ("Cellular respiration", "Cellular respiration is a set of metabolic reactions and processes that take place in the cells of organisms to convert biochemical energy from nutrients into ATP."),
        ("Magnetism", "Magnetism is a class of physical phenomena that are mediated by magnetic fields, produced by electric currents and the intrinsic magnetic moments of elementary particles."),
        ("Tectonic earthquakes", "Earthquakes are caused by the sudden release of energy in the Earth's lithosphere that creates seismic waves, most commonly resulting from movement along geological faults."),
        ("Protein folding", "Protein folding is the physical process by which a protein chain is translated to its native three-dimensional structure, typically a biologically functional conformation."),
        ("Sound waves", "Sound is a vibration that propagates as an acoustic wave through a transmission medium such as a gas, liquid, or solid."),
        ("Volcanoes", "A volcano is a rupture in the crust of a planetary-mass object that allows hot lava, volcanic ash, and gases to escape from a magma chamber below the surface."),
        ("Semiconductors", "A semiconductor is a material which has an electrical conductivity value falling between that of a conductor and an insulator."),
        ("Tides", "Tides are the rise and fall of sea levels caused by the combined effects of the gravitational forces exerted by the Moon and the Sun and the rotation of the Earth."),
        ("Photovoltaics", "A solar cell, or photovoltaic cell, is an electronic device that converts the energy of light directly into electricity by the photovoltaic effect."),
        ("Lymphatic system", "The lymphatic system is a network of tissues, organs, and vessels that help to maintain the body's fluid balance and defend against infections."),
        ("Superconductivity", "Superconductivity is a set of physical properties observed in certain materials where electrical resistance vanishes and magnetic flux fields are expelled."),
        ("Enzyme catalysis", "Enzymes are biological catalysts that accelerate chemical reactions in living organisms by lowering the activation energy required for the reaction."),
        ("Atmospheric layers", "Earth's atmosphere is divided into five main layers: the troposphere, stratosphere, mesosphere, thermosphere, and exosphere, each defined by temperature changes."),
        ("Genetics", "Genetics is the study of genes, genetic variation, and heredity in organisms, examining how traits are passed from parents to offspring through DNA."),
        ("Thermodynamics", "Thermodynamics is a branch of physics that deals with heat, work, and temperature, and their relation to energy, entropy, and the physical properties of matter."),
        ("Photons", "A photon is an elementary particle that is the quantum of the electromagnetic field, including electromagnetic radiation such as light and radio waves."),
        ("Ecosystems", "An ecosystem is a community of living organisms in conjunction with the nonliving components of their environment, interacting as a system."),
        ("Blood circulation", "The circulatory system is an organ system that permits blood to circulate and transport nutrients, oxygen, carbon dioxide, and hormones to and from cells in the body."),
        ("Glaciers", "A glacier is a persistent body of dense ice that is constantly moving under its own weight, formed where the accumulation of snow exceeds its ablation over many years."),
        ("Nuclear fusion", "Nuclear fusion is a reaction in which two or more atomic nuclei are combined to form one or more different atomic nuclei and subatomic particles."),
        ("Memory formation", "Memory formation involves the encoding, storage, and retrieval of information in the brain through changes in synaptic strength and neural connectivity."),
        ("Coral reefs", "Coral reefs are underwater ecosystems characterized by reef-building corals, formed of colonies of coral polyps held together by calcium carbonate."),
        ("Electromagnetism", "Electromagnetism is a fundamental interaction that occurs between particles with electric charge via electromagnetic fields."),
        ("Respiration", "Respiration is the biochemical process in which cells of an organism obtain energy by combining oxygen and glucose, resulting in the release of carbon dioxide and water."),
        ("Continental drift", "Continental drift is the hypothesis that the Earth's continents have moved over geologic time relative to each other, appearing to have drifted across the ocean bed."),
        ("Stem cells", "Stem cells are undifferentiated biological cells that can differentiate into specialized cells and can divide to produce more stem cells."),
    ],
    'argumentative': [
        ("French Revolution", "The French Revolution of 1789 was the product of decades of social inequality, economic crisis, and political dysfunction."),
        ("Universal basic income", "Universal basic income has been proposed as a solution to growing economic inequality and job displacement due to automation."),
        ("Nuclear energy", "Nuclear energy remains one of the most controversial power sources, with strong arguments both for and against its expansion."),
        ("Social media", "Social media platforms have fundamentally transformed how humans communicate, but their impact on mental health and democracy is increasingly questioned."),
        ("Space exploration", "Investment in space exploration has been debated since the Apollo program, with critics arguing the funds could be better spent on Earth."),
        ("Genetic engineering", "The development of CRISPR gene editing technology has reignited debates about the ethics of modifying human DNA."),
        ("Free trade", "Free trade agreements have lifted millions out of poverty globally, but have also contributed to job losses in developed nations."),
        ("Capital punishment", "Capital punishment remains legal in many countries despite growing evidence questioning both its morality and its effectiveness as a deterrent."),
        ("AI regulation", "The rapid advancement of artificial intelligence has prompted calls for government regulation to prevent misuse and ensure safety."),
        ("Animal testing", "Animal testing has been instrumental in medical breakthroughs but raises profound ethical questions about animal welfare and rights."),
        ("Immigration", "Immigration policies reflect deep tensions between economic pragmatism, humanitarian obligations, and cultural identity."),
        ("Gun control", "The debate over gun control in the United States remains one of the most politically divisive issues in American public life."),
        ("Drug legalization", "The legalization of recreational drugs has gained political momentum in many countries, driven by arguments about personal freedom and failed prohibition."),
        ("Privacy vs security", "The tension between individual privacy and national security has intensified with the proliferation of digital surveillance technologies."),
        ("Education reform", "Traditional education systems face mounting criticism for failing to prepare students for a rapidly changing economy and labor market."),
        ("Wealth inequality", "The widening gap between the richest and poorest members of society has become one of the defining political issues of the twenty-first century."),
        ("Factory farming", "Industrial animal agriculture produces food at enormous scale but at significant cost to animal welfare, public health, and the environment."),
        ("Electoral systems", "First-past-the-post voting systems have been criticized for producing unrepresentative outcomes and entrenching two-party dominance."),
        ("Intellectual property", "The current intellectual property system was designed for a pre-digital world and increasingly conflicts with how information is created and shared."),
        ("Vaccine mandates", "Mandatory vaccination policies have proven effective at controlling infectious diseases but face opposition on grounds of bodily autonomy."),
        ("Remote work", "The shift to remote work during the pandemic revealed both significant productivity benefits and concerning effects on social cohesion and mental health."),
        ("Censorship", "Government censorship of media and expression remains widespread globally, justified by appeals to public safety, morality, or national security."),
        ("Homeschooling", "The growth of homeschooling has been driven by dissatisfaction with public education, but critics question its effects on socialization and academic rigor."),
        ("Military spending", "Military spending accounts for a large share of government budgets in many nations, raising questions about opportunity costs and the drivers of arms races."),
        ("Renewable mandates", "Mandatory renewable energy targets have accelerated the clean energy transition but created economic disruption in fossil fuel dependent communities."),
        ("Universal healthcare", "Universal healthcare systems provide coverage to all citizens but face challenges of cost control, wait times, and quality variation."),
        ("Digital currency", "Central bank digital currencies promise to modernize monetary systems but raise concerns about financial surveillance and the elimination of cash."),
        ("Gig economy", "The gig economy has provided flexibility for millions of workers while eroding traditional labor protections and benefits."),
        ("Autonomous weapons", "The development of autonomous weapons systems raises fundamental questions about accountability, proportionality, and the role of human judgment in warfare."),
        ("Standardized testing", "Standardized tests remain the primary tool for measuring educational achievement despite persistent criticisms about equity, validity, and teaching to the test."),
        ("Urban sprawl", "Suburban expansion has provided affordable housing for millions but at significant environmental, social, and infrastructure costs."),
        ("Social credit systems", "Social credit scoring systems, as implemented in China, represent a novel approach to governance that Western observers have widely criticized as authoritarian."),
        ("Meat consumption", "Global meat consumption continues to rise despite evidence linking it to climate change, deforestation, and increased health risks."),
        ("Open borders", "Open borders advocates argue that freedom of movement is a fundamental human right, while opponents cite economic and security concerns."),
        ("Term limits", "Term limits for elected officials aim to prevent the entrenchment of power but may also reduce institutional expertise and legislative effectiveness."),
        ("Carbon tax", "A carbon tax is widely considered the most economically efficient tool for reducing greenhouse gas emissions, but implementation faces fierce political opposition."),
        ("Student debt", "The student debt crisis in the United States has reached 1.7 trillion dollars, sparking debates about the fundamental model of higher education financing."),
        ("Nuclear proliferation", "The spread of nuclear weapons capability to additional states remains one of the gravest threats to international security."),
        ("Zoning laws", "Restrictive zoning regulations have been identified as a major contributor to housing shortages and rising home prices in major cities."),
        ("Foreign aid", "Foreign aid programs remain controversial, with supporters citing moral obligation and critics questioning their effectiveness and unintended consequences."),
    ],
    'narrative': [
        ("Penicillin discovery", "In 1928, Alexander Fleming returned to his laboratory at St Mary's Hospital after a vacation to find something unexpected."),
        ("Moon landing", "On July 20, 1969, Neil Armstrong descended the ladder of the lunar module Eagle and stepped onto the surface of the Moon."),
        ("Pompeii", "On August 24, 79 AD, the citizens of Pompeii went about their daily lives unaware that Mount Vesuvius was about to erupt."),
        ("Turing's machine", "In 1936, a young mathematician named Alan Turing published a paper that would lay the foundation for modern computing."),
        ("Marie Curie", "In 1897, Marie Curie began her doctoral research on uranium rays, a decision that would change the course of science."),
        ("Rosetta Stone", "When Napoleon's soldiers discovered the Rosetta Stone in 1799, they could not have known it would unlock the secrets of Egyptian hieroglyphics."),
        ("Darwin's finches", "When Charles Darwin arrived at the Galápagos Islands in 1835, he noticed something peculiar about the local finches."),
        ("Manhattan Project", "In August 1939, Albert Einstein signed a letter to President Roosevelt warning that Germany might be developing an atomic bomb."),
        ("Gutenberg's press", "In the 1440s, a goldsmith named Johannes Gutenberg began experimenting with movable metal type in Mainz, Germany."),
        ("Double helix", "In early 1953, James Watson and Francis Crick were racing to determine the structure of DNA at Cambridge University."),
        ("Fall of Constantinople", "On May 29, 1453, Sultan Mehmed II's Ottoman forces breached the walls of Constantinople after a siege lasting nearly two months."),
        ("Galileo's telescope", "In 1609, Galileo Galilei pointed a newly improved telescope toward the night sky and began making observations that would challenge centuries of accepted wisdom."),
        ("Boston Tea Party", "On the night of December 16, 1773, a group of American colonists disguised as Mohawk Indians boarded three British ships in Boston Harbor."),
        ("Chernobyl", "At 1:23 AM on April 26, 1986, an explosion ripped through Reactor No. 4 at the Chernobyl Nuclear Power Plant in Ukraine."),
        ("First flight", "On December 17, 1903, on the wind-swept sand dunes of Kitty Hawk, North Carolina, Orville Wright lay prone on the lower wing of a fragile biplane."),
        ("Assassination of Caesar", "On the Ides of March, 44 BC, Julius Caesar walked into the Theatre of Pompey to attend a meeting of the Roman Senate."),
        ("Smallpox eradication", "In 1967, the World Health Organization launched an ambitious campaign to eradicate smallpox, a disease that had killed hundreds of millions throughout history."),
        ("Enigma machine", "In 1939, Polish mathematicians smuggled the secret of the German Enigma cipher machine to the British, setting in motion one of the war's most important intelligence operations."),
        ("Lewis and Clark", "On May 14, 1804, Meriwether Lewis and William Clark departed from Camp Dubois near St. Louis on an expedition that would take them across the uncharted American West."),
        ("Magellan's voyage", "On September 20, 1519, Ferdinand Magellan set sail from Spain with five ships and approximately 270 men, aiming to find a western route to the Spice Islands."),
        ("Archimedes eureka", "According to legend, Archimedes was lowering himself into a public bath in Syracuse when he noticed the water level rising in proportion to his submerged body."),
        ("Krakatoa", "On August 27, 1883, the volcanic island of Krakatoa in the Sunda Strait between Java and Sumatra erupted with a force that was heard nearly 5,000 kilometers away."),
        ("Newton's apple", "In 1666, a young Isaac Newton retreated to his family's farm in Woolsthorpe during the Great Plague, where he would have one of the most productive intellectual periods in scientific history."),
        ("D-Day", "In the early hours of June 6, 1944, more than 156,000 Allied troops crossed the English Channel in the largest amphibious invasion in military history."),
        ("Titanic sinking", "At 11:40 PM on April 14, 1912, the lookout in the crow's nest of the RMS Titanic spotted an iceberg directly in the ship's path."),
        ("Apollo 13", "On April 13, 1970, astronaut Jack Swigert radioed Houston with five words that would become famous: 'Houston, we've had a problem.'"),
        ("Pasteur's experiment", "In 1859, Louis Pasteur designed an elegant experiment using swan-neck flasks that would finally disprove the theory of spontaneous generation."),
        ("Fall of Berlin Wall", "On the evening of November 9, 1989, East German spokesman Günter Schabowski made an announcement at a press conference that would accidentally change the course of history."),
        ("Voyager launch", "On September 5, 1977, NASA launched Voyager 1 from Cape Canaveral, carrying a golden record containing sounds and images of Earth intended for any extraterrestrial civilization that might find it."),
        ("Silk Road", "Around 130 BC, the Chinese explorer Zhang Qian returned to the court of Emperor Wu after thirteen years of captivity and travel, bringing news of unknown civilizations to the west."),
        ("Hiroshima", "At 8:15 AM on August 6, 1945, the crew of the Enola Gay released a single bomb over the Japanese city of Hiroshima."),
        ("First vaccine", "In May 1796, an English country doctor named Edward Jenner took material from a cowpox sore on the hand of a milkmaid and scratched it into the arm of an eight-year-old boy."),
        ("Waterloo", "On the morning of June 18, 1815, Napoleon Bonaparte surveyed the rain-soaked battlefield near the Belgian village of Waterloo, confident that the day would bring him victory."),
        ("Tutankhamun's tomb", "On November 26, 1922, archaeologist Howard Carter peered through a small hole in a sealed doorway in the Valley of the Kings and saw 'wonderful things.'"),
        ("Great Fire of London", "In the early hours of September 2, 1666, a fire broke out in a bakery on Pudding Lane in the City of London."),
        ("Challenger disaster", "On the morning of January 28, 1986, millions of Americans watched live television coverage as the Space Shuttle Challenger lifted off from Cape Canaveral."),
        ("Copernicus", "In 1543, as he lay on his deathbed in Frombork, Nicolaus Copernicus was presented with a printed copy of his revolutionary work, De Revolutionibus Orbium Coelestium."),
        ("Panama Canal", "In 1881, the French began the most ambitious engineering project in history: cutting a canal through the mountainous spine of Panama to connect the Atlantic and Pacific oceans."),
        ("Black Death", "In October 1347, twelve Genoese trading ships docked at the Sicilian port of Messina, carrying sailors who were either dead or gravely ill with a mysterious disease."),
        ("Mendel's peas", "Between 1856 and 1863, an Augustinian friar named Gregor Mendel cultivated and tested roughly 29,000 pea plants in the monastery garden at Brno, meticulously recording patterns of inheritance."),
    ],
    'comparative': [
        ("Renewable energy", "Renewable energy sources like solar and wind power have seen dramatic cost reductions over the past decade, making them increasingly competitive with fossil fuels."),
        ("Democracy vs autocracy", "Democratic and autocratic systems of government differ fundamentally in how power is distributed, exercised, and constrained."),
        ("Classical vs quantum", "Classical physics and quantum mechanics describe the same universe but operate under fundamentally different assumptions."),
        ("Cats vs dogs", "Dogs and cats have been domesticated for thousands of years, but they differ profoundly in their social behavior, evolutionary history, and relationship with humans."),
        ("East vs West", "Eastern and Western philosophical traditions have developed largely independently, arriving at different conclusions about consciousness, ethics, and reality."),
        ("Urban vs rural", "Urban and rural lifestyles offer distinct advantages and challenges that reflect deep differences in community structure and opportunity."),
        ("Digital vs analog", "The transition from analog to digital technology has transformed nearly every aspect of modern life."),
        ("Rome vs Greece", "Ancient Rome and Greece both shaped Western civilization profoundly, but their contributions were fundamentally different in character."),
        ("RNA vs DNA", "RNA and DNA are both nucleic acids that store genetic information, but they differ in structure, function, and stability."),
        ("Capitalism vs socialism", "Capitalism and socialism represent fundamentally different approaches to organizing economic activity and distributing wealth."),
        ("Prose vs poetry", "Prose and poetry are the two fundamental modes of literary expression, each with distinct conventions, rhythms, and effects on the reader."),
        ("Bacteria vs viruses", "Bacteria and viruses are both microscopic agents of disease, but they differ fundamentally in their biology, treatment, and evolutionary strategies."),
        ("Introvert vs extrovert", "Introversion and extroversion represent two ends of a spectrum describing how individuals derive and expend social energy."),
        ("Fresh vs salt water", "Freshwater and saltwater ecosystems support vastly different communities of organisms despite sharing many fundamental ecological principles."),
        ("Monarchy vs republic", "Monarchies and republics represent two of the oldest forms of government, each with distinctive strengths and vulnerabilities."),
        ("Aerobic vs anaerobic", "Aerobic and anaerobic exercise differ in their energy systems, physiological effects, and health benefits."),
        ("Impressionism vs realism", "Impressionism and Realism represent contrasting artistic philosophies that emerged in nineteenth-century France."),
        ("Deductive vs inductive", "Deductive and inductive reasoning represent two complementary approaches to logical inference, each with distinct strengths and limitations."),
        ("AC vs DC", "Alternating current and direct current differ in how electrical charge flows through a conductor, with important practical implications for power distribution."),
        ("Mitosis vs meiosis", "Mitosis and meiosis are both forms of cell division, but they serve different biological purposes and produce genetically distinct outcomes."),
        ("Olympic vs Paralympic", "The Olympic and Paralympic Games share a common venue and schedule but differ in their history, classification systems, and cultural significance."),
        ("Tropical vs temperate", "Tropical and temperate forests differ dramatically in their species diversity, seasonal patterns, and ecological dynamics."),
        ("Ethics vs morals", "Ethics and morals are frequently used interchangeably, but they refer to distinct concepts with different philosophical foundations."),
        ("Fission vs fusion", "Nuclear fission and fusion both release enormous amounts of energy from atomic nuclei, but through opposite processes."),
        ("Herbs vs spices", "Herbs and spices are both used to flavor food, but they come from different parts of plants and have distinct culinary traditions."),
        ("Comedy vs tragedy", "Comedy and tragedy are the two foundational dramatic forms, defined by Aristotle and still shaping storytelling today."),
        ("Weather vs climate", "Weather and climate describe atmospheric conditions at fundamentally different time scales, yet are frequently confused in public discourse."),
        ("Arteries vs veins", "Arteries and veins are both blood vessels, but they differ in structure, function, and the direction of blood flow."),
        ("Sympathy vs empathy", "Sympathy and empathy are related but distinct emotional responses to another person's situation."),
        ("Hardware vs software", "Hardware and software are the two complementary components of any computing system, each dependent on the other."),
        ("Crocodiles vs alligators", "Crocodiles and alligators are both large aquatic reptiles, but they belong to different families and differ in morphology and behavior."),
        ("Stocks vs bonds", "Stocks and bonds are the two primary asset classes in investment portfolios, offering fundamentally different risk-return profiles."),
        ("Primary vs secondary", "Primary and secondary sources serve different roles in research, with distinct standards of reliability and interpretation."),
        ("Asteroids vs comets", "Asteroids and comets are both remnants of the early solar system, but they differ in composition, origin, and orbital characteristics."),
        ("Liberalism vs conservatism", "Liberalism and conservatism represent the two dominant political ideologies in Western democracies, differing in their views on change, tradition, and the role of the state."),
        ("Bees vs wasps", "Bees and wasps are closely related insects that are frequently confused, but they differ significantly in behavior, diet, and ecological role."),
        ("Acid vs base", "Acids and bases are two fundamental categories of chemical compounds, defined by their behavior in aqueous solution."),
        ("Memory vs storage", "Memory and storage serve complementary roles in computing, differing in speed, volatility, and capacity."),
        ("Granite vs basalt", "Granite and basalt are the two most common igneous rocks, formed under different conditions with distinct mineral compositions."),
        ("Hypothesis vs theory", "In scientific usage, a hypothesis and a theory represent different stages of scientific understanding, though the distinction is often misunderstood."),
    ],
    'causal': [
        ("Climate change", "Climate change refers to the long-term shift in global temperatures and weather patterns, primarily driven by human activities."),
        ("Obesity epidemic", "The global obesity epidemic has been driven by a complex interplay of biological, environmental, and socioeconomic factors."),
        ("World War I", "The outbreak of World War I in 1914 resulted from a chain of events triggered by the assassination of Archduke Franz Ferdinand."),
        ("Coral reef decline", "Coral reef ecosystems worldwide are in decline due to a combination of human activities and environmental changes."),
        ("Antibiotic resistance", "Antibiotic resistance has emerged as one of the most serious public health threats of the twenty-first century."),
        ("Deforestation", "Deforestation in tropical regions has accelerated dramatically over the past century, with cascading effects on climate and biodiversity."),
        ("Financial crisis", "The 2008 global financial crisis was triggered by the collapse of the US housing market and the failure of major financial institutions."),
        ("Language extinction", "Languages are going extinct at an unprecedented rate, with approximately one language dying every two weeks."),
        ("Insect decline", "Global insect populations have declined by an estimated forty percent over the past several decades, with alarming implications for ecosystems."),
        ("Brain drain", "The emigration of highly skilled workers from developing to developed countries has significant economic and social consequences."),
        ("Soil erosion", "Soil erosion removes the most fertile topsoil layer at rates far exceeding natural replacement, threatening agricultural productivity worldwide."),
        ("Ocean acidification", "The absorption of atmospheric carbon dioxide by the oceans is causing a measurable decrease in ocean pH, with profound effects on marine life."),
        ("Opioid crisis", "The opioid epidemic in the United States began in the 1990s when pharmaceutical companies aggressively marketed prescription painkillers as safe and non-addictive."),
        ("Urbanization", "Rapid urbanization is transforming societies worldwide as populations shift from rural areas to cities at an unprecedented rate."),
        ("Water scarcity", "Water scarcity affects more than two billion people globally and is driven by population growth, climate change, and inefficient water management."),
        ("Biodiversity loss", "The current rate of species extinction is estimated to be one hundred to one thousand times higher than natural background rates."),
        ("Inflation", "Inflation occurs when the general price level of goods and services rises over time, reducing the purchasing power of money."),
        ("Sleep deprivation", "Chronic sleep deprivation has been linked to a wide range of health problems, including cardiovascular disease, diabetes, and cognitive impairment."),
        ("Desertification", "Desertification is the degradation of land in arid and semi-arid regions, caused primarily by human activities and climatic variations."),
        ("Refugee crisis", "Forced displacement has reached historic levels, with over one hundred million people displaced worldwide by conflict, persecution, and climate change."),
        ("Light pollution", "Artificial light at night has increased globally by at least forty-nine percent over the past twenty-five years, disrupting ecosystems and human health."),
        ("Microplastic pollution", "Microplastics have been found in virtually every environment on Earth, from the deepest ocean trenches to the highest mountain peaks."),
        ("Aging population", "The proportion of elderly people in the global population is increasing rapidly, driven by declining birth rates and rising life expectancy."),
        ("Permafrost thaw", "Permafrost covers approximately twenty-five percent of the Northern Hemisphere's land surface and is thawing at an accelerating rate due to global warming."),
        ("Mental health crisis", "Mental health disorders have increased dramatically worldwide, with depression and anxiety being the most common conditions affecting hundreds of millions."),
        ("Bee colony collapse", "Colony collapse disorder, in which worker bees abruptly disappear from a hive, has been linked to pesticide exposure, parasites, and habitat loss."),
        ("Topsoil depletion", "The world has lost approximately one-third of its arable topsoil in the past forty years due to erosion, compaction, and chemical degradation."),
        ("Digital divide", "The gap between those with access to modern information technology and those without continues to shape economic opportunity and social mobility."),
        ("Coral bleaching", "Mass coral bleaching events have become five times more frequent since the 1980s, driven primarily by rising ocean temperatures."),
        ("Air pollution deaths", "Ambient air pollution is responsible for an estimated four point two million premature deaths annually, primarily from heart disease, stroke, and lung cancer."),
        ("Groundwater depletion", "Aquifers around the world are being depleted faster than they can be naturally recharged, threatening water supplies for billions of people."),
        ("Wildfire increase", "The frequency and severity of wildfires have increased dramatically in many regions, driven by climate change, drought, and land management practices."),
        ("Antibiotic overuse", "The overuse of antibiotics in both human medicine and animal agriculture has accelerated the evolution of drug-resistant bacteria."),
        ("Income stagnation", "Real wages for middle and lower-income workers have stagnated in many developed countries over the past four decades despite rising productivity."),
        ("Plastic waste", "Annual global plastic production has exceeded four hundred million tons, with less than ten percent being recycled."),
        ("Pollinator decline", "Wild pollinator populations have declined significantly across North America and Europe, threatening crop yields and ecosystem health."),
        ("Methane emissions", "Methane concentrations in the atmosphere have more than doubled since pre-industrial times, contributing approximately thirty percent of current global warming."),
        ("Food waste", "Roughly one-third of all food produced globally for human consumption is lost or wasted each year, amounting to approximately 1.3 billion tons."),
        ("Coastal erosion", "Rising sea levels and increased storm intensity are accelerating coastal erosion, threatening communities and infrastructure worldwide."),
        ("Heat waves", "Extreme heat events have become more frequent and intense due to climate change, causing tens of thousands of excess deaths annually."),
    ],
}


# ============================================================
# RUN
# ============================================================

total_seeds = sum(len(v) for v in SEEDS.values())
print(f"\n{'=' * 80}")
print(f"SUPERSATURATION TAXONOMY v2")
print(f"{total_seeds} seeds across {len(SEEDS)} discourse types")
print(f"Threshold: {SAT_THRESHOLD}, max branches: 12")
print(f"{'=' * 80}")

all_results = []
type_traces = {dtype: [] for dtype in SEEDS}
t_start = time.time()

for dtype, seeds in SEEDS.items():
    print(f"\n{'═' * 60}")
    print(f"DISCOURSE TYPE: {dtype.upper()} ({len(seeds)} seeds)")
    print(f"{'═' * 60}")

    for si, (label, seed_text) in enumerate(seeds):
        t0 = time.time()
        result = grow_with_trace(seed_text, max_branches=12, max_tokens=512)
        elapsed = time.time() - t0

        trace_sats = [t['sat_total'] for t in result['trace'] if t['status'] == 'ok']
        trace_struct = [t['sat_struct'] for t in result['trace'] if t['status'] == 'ok']
        trace_sem = [t['sat_sem'] for t in result['trace'] if t['status'] == 'ok']

        # Classify decay shape
        if len(trace_sats) >= 2:
            diffs = [trace_sats[i+1] - trace_sats[i] for i in range(len(trace_sats)-1)]
            n_rising = sum(1 for d in diffs if d > 0.02)
            n_falling = sum(1 for d in diffs if d < -0.02)
            n_flat = len(diffs) - n_rising - n_falling

            if n_rising == 0 and n_falling > 0:
                shape = 'monotone_decay'
            elif n_rising > 0 and n_falling > 0:
                shape = 'oscillating'
            elif n_flat == len(diffs):
                shape = 'flat'
            elif n_rising > n_falling:
                shape = 'rising'
            else:
                shape = 'mixed'
        elif len(trace_sats) == 1:
            shape = 'single_branch'
        else:
            shape = 'empty'

        entry = {
            'label': label,
            'dtype': dtype,
            'seed': seed_text[:80],
            'trace': result['trace'],
            'trace_sats': trace_sats,
            'trace_struct': trace_struct,
            'trace_sem': trace_sem,
            'shape': shape,
            'n_branches': result['n_branches'],
            'total_tokens': result['total_tokens'],
            'elapsed': round(elapsed, 1),
            'final_text': result['final_text'][:200],
        }
        all_results.append(entry)
        type_traces[dtype].append(trace_sats)

        sat_str = ' → '.join(f'{s:.3f}' for s in trace_sats)
        done = len(all_results)
        eta_min = (elapsed * (total_seeds - done)) / 60
        print(f"  [{done}/{total_seeds}] {label:25s}: {result['n_branches']:2d}br, "
              f"{result['total_tokens']:3d}tok, shape={shape:15s} | {sat_str} "
              f"({elapsed:.0f}s, ETA {eta_min:.0f}m)")

        # Save incremental results every 10 seeds
        if done % 10 == 0:
            with open('/data/sat_taxonomy_v2_partial.json', 'w') as f:
                json.dump({'results': all_results, 'done': done, 'total': total_seeds}, f)


# ============================================================
# ANALYSIS
# ============================================================

elapsed_total = (time.time() - t_start) / 60
print(f"\n{'=' * 80}")
print(f"ANALYSIS (completed in {elapsed_total:.0f} minutes)")
print(f"{'=' * 80}")

# 1. Shape distribution
print("\n1. DECAY SHAPE DISTRIBUTION")
print(f"{'Type':15s} | {'monotone':>8s} | {'oscillat':>8s} | {'flat':>8s} | "
      f"{'rising':>8s} | {'single':>8s} | {'mixed':>8s} | {'empty':>8s} | "
      f"{'multi%':>6s}")
print("-" * 100)
for dtype in SEEDS:
    shapes = [r['shape'] for r in all_results if r['dtype'] == dtype]
    counts = Counter(shapes)
    total = len(shapes)
    multi = total - counts.get('single_branch', 0) - counts.get('empty', 0)
    print(f"{dtype:15s} | {counts.get('monotone_decay',0):8d} | "
          f"{counts.get('oscillating',0):8d} | {counts.get('flat',0):8d} | "
          f"{counts.get('rising',0):8d} | {counts.get('single_branch',0):8d} | "
          f"{counts.get('mixed',0):8d} | {counts.get('empty',0):8d} | "
          f"{100*multi/total:5.1f}%")

# 2. Average initial supersaturation
print("\n2. AVERAGE INITIAL SUPERSATURATION")
for dtype in SEEDS:
    init_sats = [r['trace_sats'][0] for r in all_results
                 if r['dtype'] == dtype and r['trace_sats']]
    if init_sats:
        print(f"  {dtype:15s}: {np.mean(init_sats):.3f} ± {np.std(init_sats):.3f} "
              f"(range {min(init_sats):.3f}–{max(init_sats):.3f})")

# 3. Average branch count
print("\n3. AVERAGE BRANCH COUNT")
for dtype in SEEDS:
    branches = [r['n_branches'] for r in all_results if r['dtype'] == dtype]
    print(f"  {dtype:15s}: {np.mean(branches):.1f} ± {np.std(branches):.1f} "
          f"(max {max(branches)})")

# 4. Structural/semantic ratio (all branches, not just first)
print("\n4. STRUCTURAL/SEMANTIC RATIO (across all branches)")
for dtype in SEEDS:
    ratios = []
    for r in all_results:
        if r['dtype'] == dtype:
            for s_struct, s_sem in zip(r['trace_struct'], r['trace_sem']):
                if s_sem > 0.01:
                    ratios.append(s_struct / s_sem)
    if ratios:
        print(f"  {dtype:15s}: {np.mean(ratios):.3f} ± {np.std(ratios):.3f} "
              f"(n={len(ratios)})")

# 5. Decay rate (multi-branch only)
print("\n5. DECAY RATE (linear slope, multi-branch traces only)")
for dtype in SEEDS:
    slopes = []
    for r in all_results:
        if r['dtype'] == dtype and len(r['trace_sats']) >= 3:
            x = list(range(len(r['trace_sats'])))
            slope = np.polyfit(x, r['trace_sats'], 1)[0]
            slopes.append(slope)
    if slopes:
        print(f"  {dtype:15s}: {np.mean(slopes):+.4f} ± {np.std(slopes):.4f} "
              f"(n={len(slopes)})")

# 6. Multi-branch trace examples
print("\n6. NOTABLE MULTI-BRANCH TRACES")
for r in sorted(all_results, key=lambda x: -x['n_branches'])[:10]:
    sat_str = ' → '.join(f'{s:.3f}' for s in r['trace_sats'])
    print(f"  {r['label']:25s} ({r['dtype']:12s}): {r['n_branches']:2d}br, "
          f"shape={r['shape']:15s} | {sat_str}")


# ============================================================
# SAVE
# ============================================================

outpath = '/data/sat_taxonomy_v2_results.json'
with open(outpath, 'w') as f:
    json.dump({
        'experiment': 'supersaturation_taxonomy_v2',
        'model': model_name,
        'threshold': SAT_THRESHOLD,
        'n_seeds': len(all_results),
        'discourse_types': list(SEEDS.keys()),
        'elapsed_minutes': round(elapsed_total, 1),
        'results': all_results,
    }, f, indent=2, default=str)
print(f"\nSaved to {outpath}")
