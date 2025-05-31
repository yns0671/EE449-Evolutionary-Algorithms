
import cv2
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import euclidean

# Gene class to represent a triangle
class Gene:
    def __init__(self, width, height):
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: width={width}, height={height}")
        # Initialize vertices and color
        self.vertices = [(np.random.randint(0, width), np.random.randint(0, height)) for _ in range(3)]
        self.color = [int(np.random.randint(0, 256)) for _ in range(3)] + [np.random.uniform(0, 1)]
        self.area = self.calculate_area()
        
        # Ensure triangle is within image bounds
        while not self.is_within_bounds(width, height):
            self.vertices = [(np.random.randint(0, width), np.random.randint(0, height)) for _ in range(3)]
            self.color = [int(np.random.randint(0, 256)) for _ in range(3)] + [np.random.uniform(0, 1)]
            self.area = self.calculate_area()

    def calculate_area(self):
        # Calculate triangle area using the shoelace formula
        x1, y1 = self.vertices[0]
        x2, y2 = self.vertices[1]
        x3, y3 = self.vertices[2]
        return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    def is_within_bounds(self, width, height):
        # Check if at least one vertex is within image bounds
        for x, y in self.vertices:
            if 0 <= x < width and 0 <= y < height:
                return True
        return False

    def mutate(self, width, height, mutation_type, mutation_prob):
        if np.random.random() < mutation_prob:
            if mutation_type == 'unguided':
                self.vertices = [(np.random.randint(0, width), np.random.randint(0, height)) for _ in range(3)]
                self.color = [int(np.random.randint(0, 256)) for _ in range(3)] + [np.random.uniform(0, 1)]
            elif mutation_type == 'guided':
                for i in range(3):
                    self.vertices[i] = (
                        np.clip(self.vertices[i][0] + np.random.randint(-width//4, width//4), 0, width-1),
                        np.clip(self.vertices[i][1] + np.random.randint(-height//4, height//4), 0, height-1)
                    )
                self.color = [
                    int(np.clip(self.color[0] + np.random.randint(-64, 65), 0, 255)),
                    int(np.clip(self.color[1] + np.random.randint(-64, 65), 0, 255)),
                    int(np.clip(self.color[2] + np.random.randint(-64, 65), 0, 255)),
                    np.clip(self.color[3] + np.random.uniform(-0.25, 0.25), 0, 1)
                ]
            self.area = self.calculate_area()

# Individual class to represent a chromosome
class Individual:
    def __init__(self, num_genes, width, height):
        self.genes = [Gene(width, height) for _ in range(num_genes)]
        self.genes.sort(key=lambda x: x.area, reverse=True)
        self.fitness = None
        self.raw_fitness = None
        self.shared_fitness = None

    def draw_image(self, shape):
        # Initialize white image
        image = np.ones(shape, dtype=np.uint8) * 255
        for gene in self.genes:
            overlay = image.copy()
            pts = np.array(gene.vertices, np.int32).reshape((-1, 1, 2))
            color = tuple(int(c) for c in gene.color[:3])
            try:
                cv2.fillPoly(overlay, [pts], color)
                image = cv2.addWeighted(overlay, gene.color[3], image, 1 - gene.color[3], 0)
            except Exception as e:
                print(f"Error in cv2.fillPoly: {e}")
                print(f"Vertices: {pts}, Color: {color}")
                raise
        return image

# Population class
class Population:
    def __init__(self, num_inds, num_genes, width, height):
        self.individuals = [Individual(num_genes, width, height) for _ in range(num_inds)]
        self.current_num_genes = num_genes
        self.best_fitness = 0
        self.stagnation_count = 0
        self.mutation_prob = 0.2

    def evaluate(self, source_image, use_fitness_sharing=False):
        for ind in self.individuals:
            generated_image = ind.draw_image(source_image.shape)
            ind.raw_fitness = calculate_ssim(source_image, generated_image)
            ind.fitness = ind.raw_fitness
        if use_fitness_sharing:
            self.apply_fitness_sharing()

    def apply_fitness_sharing(self, sigma=100.0):
        for i, ind1 in enumerate(self.individuals):
            sharing_sum = 0
            for j, ind2 in enumerate(self.individuals):
                if i != j:
                    distance = self.compute_individual_distance(ind1, ind2)
                    sharing_sum += np.exp(-distance / sigma)
            ind1.shared_fitness = ind1.raw_fitness / (1 + sharing_sum)
            ind1.fitness = ind1.shared_fitness

    def compute_individual_distance(self, ind1, ind2):
        total_distance = 0
        min_genes = min(len(ind1.genes), len(ind2.genes))
        for g1, g2 in zip(ind1.genes[:min_genes], ind2.genes[:min_genes]):
            vertex_distance = sum((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2 
                                 for v1, v2 in zip(g1.vertices, g2.vertices))
            color_distance = sum((c1 - c2)**2 for c1, c2 in zip(g1.color, g2.color))
            total_distance += np.sqrt(vertex_distance + color_distance)
        return total_distance / min_genes if min_genes > 0 else 0

    def adjust_triangle_count(self, gen, max_gen, max_genes):
        progress = gen / max_gen
        target_genes = int(10 + (max_genes - 10) * progress)
        if target_genes > self.current_num_genes:
            for ind in self.individuals:
                ind.genes.extend([Gene(source_image.shape[1], source_image.shape[0])
                                  for _ in range(target_genes - len(ind.genes))])
                ind.genes.sort(key=lambda x: x.area, reverse=True)
        self.current_num_genes = target_genes

    def update_mutation_prob(self, new_best_fitness, stagnation_threshold=0.001, improvement_threshold=0.01):
        if abs(new_best_fitness - self.best_fitness) < stagnation_threshold:
            self.stagnation_count += 1
            if self.stagnation_count >= 500:
                self.mutation_prob = min(self.mutation_prob * 1.5, 0.8)
                self.stagnation_count = 0
        else:
            if new_best_fitness - self.best_fitness > improvement_threshold:
                self.mutation_prob = max(self.mutation_prob * 0.5, 0.1)
            self.stagnation_count = 0
        self.best_fitness = new_best_fitness

# SSIM implementation
def calculate_ssim(source, generated):
    c1, c2 = 6.5025, 58.5225
    ssim_sum = 0
    for k in range(3):
        S = source[:, :, k].astype(np.float32)
        G = generated[:, :, k].astype(np.float32)
        mu_s = np.mean(S)
        mu_g = np.mean(G)
        sigma_s = np.mean((S - mu_s) ** 2)
        sigma_g = np.mean((G - mu_g) ** 2)
        sigma_sg = np.mean((S - mu_s) * (G - mu_g))
        numerator = (2 * mu_s * mu_g + c1) * (2 * sigma_sg + c2)
        denominator = (mu_s ** 2 + mu_g ** 2 + c1) * (sigma_s + sigma_g + c2)
        ssim_sum += numerator / denominator
    return ssim_sum / 3

# Crossover function
def crossover(parent1, parent2, num_genes, width, height):
    child1, child2 = Individual(num_genes, width, height), Individual(num_genes, width, height)
    child1.genes, child2.genes = [], []
    for i in range(num_genes):
        if np.random.random() < 0.5:
            child1.genes.append(deepcopy(parent1.genes[i]))
            child2.genes.append(deepcopy(parent2.genes[i]))
        else:
            child1.genes.append(deepcopy(parent2.genes[i]))
            child2.genes.append(deepcopy(parent1.genes[i]))
    child1.genes.sort(key=lambda x: x.area, reverse=True)
    child2.genes.sort(key=lambda x: x.area, reverse=True)
    return child1, child2

# Tournament selection
def tournament_selection(self, tm_size, use_fitness_sharing=False):
    selected = []
    fitness_key = 'shared_fitness' if use_fitness_sharing else 'fitness'
    for _ in range(len(self.individuals)):
        tournament = np.random.choice(self.individuals, tm_size)
        best = max(tournament, key=lambda x: getattr(x, fitness_key))
        selected.append(deepcopy(best))
    return selected

# Attach method to Population class
Population.tournament_selection = tournament_selection

# Evolutionary algorithm
def evolutionary_algorithm(source_image, num_inds=20, num_genes=50, tm_size=5, frac_elites=0.2, frac_parents=0.6, mutation_prob=0.2, mutation_type='guided', num_generations=10000, use_adaptive_mutation=False, use_dynamic_triangles=False, use_fitness_sharing=False):
    global source_image  # Ensure source_image is accessible in Population methods
    width, height = source_image.shape[1], source_image.shape[0]
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid source image dimensions: width={width}, height={height}")
    population = Population(num_inds, num_genes, width, height)
    fitness_history = []
    best_images = []

    for gen in range(num_generations):
        if use_dynamic_triangles:
            population.adjust_triangle_count(gen, num_generations, num_genes)
        population.evaluate(source_image, use_fitness_sharing)
        best_ind = max(population.individuals, key=lambda x: x.raw_fitness)
        fitness_history.append(best_ind.raw_fitness)
        if use_adaptive_mutation:
            population.update_mutation_prob(best_ind.raw_fitness)
        if (gen + 1) % 1000 == 0:
            best_images.append(best_ind.draw_image(source_image.shape))
        num_elites = int(frac_elites * num_inds)
        elites = sorted(population.individuals, key=lambda x: x.fitness, reverse=True)[:num_elites]
        non_elites = population.tournament_selection(tm_size, use_fitness_sharing)
        num_parents = int(frac_parents * num_inds)
        parents = sorted(non_elites, key=lambda x: x.fitness, reverse=True)[:num_parents]
        offspring = []
        for i in range(0, num_parents, 2):
            if i + 1 < num_parents:
                child1, child2 = crossover(parents[i], parents[i+1], population.current_num_genes if use_dynamic_triangles else num_genes, width, height)
                offspring.extend([child1, child2])
        for ind in offspring:
            for gene in ind.genes:
                gene.mutate(width, height, mutation_type, population.mutation_prob if use_adaptive_mutation else mutation_prob)
        population.individuals = elites + offspring[:num_inds - num_elites]
    return fitness_history, best_images

# Run all experiments
def run_experiments(source_image):
    output_dir = "output"
    output_improvements_dir = "output_improvements"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_improvements_dir, exist_ok=True)
    
    default_params = {
        'num_inds': 20,
        'num_genes': 50,
        'tm_size': 5,
        'frac_elites': 0.2,
        'frac_parents': 0.6,
        'mutation_prob': 0.2,
        'mutation_type': 'guided',
        'num_generations': 10000
    }

    # Hyperparameter experiments
    param_configs = {
        'num_inds': [5, 10, 20, 50, 75],
        'num_genes': [10, 25, 50, 100, 150],
        'tm_size': [2, 5, 10, 20],
        'frac_elites': [0.05, 0.2, 0.4],
        'frac_parents': [0.2, 0.4, 0.6, 0.8],
        'mutation_prob': [0.1, 0.2, 0.5, 0.8],
        'mutation_type': ['guided', 'unguided']
    }

    for param, values in param_configs.items():
        for value in values:
            print(f"Running baseline experiment with {param}={value}")
            params = default_params.copy()
            params[param] = value
            fitness_history, best_images = evolutionary_algorithm(source_image, **params)
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(fitness_history) + 1), fitness_history)
            plt.title(f"Fitness over Generations ({param}={value})")
            plt.xlabel("Generation")
            plt.ylabel("SSIM Fitness")
            plt.savefig(os.path.join(output_dir, f"fitness_full_{param}_{value}.png"))
            plt.close()
            plt.figure(figsize=(10, 5))
            plt.plot(range(1000, len(fitness_history) + 1), fitness_history[999:])
            plt.title(f"Fitness from Gen 1000 ({param}={value})")
            plt.xlabel("Generation")
            plt.ylabel("SSIM Fitness")
            plt.savefig(os.path.join(output_dir, f"fitness_1000_{param}_{value}.png"))
            plt.close()
            for i, img in enumerate(best_images):
                cv2.imwrite(os.path.join(output_dir, f"best_image_gen_{(i+1)*1000}_{param}_{value}.png"), img)

    # Baseline experiment for improvements
    print("Running baseline experiment for improvements")
    fitness_history, best_images = evolutionary_algorithm(source_image, **default_params)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(fitness_history) + 1), fitness_history)
    plt.title("Fitness over Generations (Baseline)")
    plt.xlabel("Generation")
    plt.ylabel("SSIM Fitness")
    plt.savefig(os.path.join(output_improvements_dir, "fitness_full_baseline.png"))
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.plot(range(1000, len(fitness_history) + 1), fitness_history[999:])
    plt.title("Fitness from Gen 1000 (Baseline)")
    plt.xlabel("Generation")
    plt.ylabel("SSIM Fitness")
    plt.savefig(os.path.join(output_improvements_dir, "fitness_1000_baseline.png"))
    plt.close()
    for i, img in enumerate(best_images):
        cv2.imwrite(os.path.join(output_improvements_dir, f"best_image_gen_{(i+1)*1000}_baseline.png"), img)

    # Improvement experiments
    improvements = [
        ('adaptive_mutation', {'use_adaptive_mutation': True, 'use_dynamic_triangles': False, 'use_fitness_sharing': False}),
        ('dynamic_triangles', {'use_adaptive_mutation': False, 'use_dynamic_triangles': True, 'use_fitness_sharing': False}),
        ('fitness_sharing', {'use_adaptive_mutation': False, 'use_dynamic_triangles': False, 'use_fitness_sharing': True})
    ]

    for improvement_name, improvement_params in improvements:
        print(f"Running experiment with improvement: {improvement_name}")
        params = default_params.copy()
        params.update(improvement_params)
        fitness_history, best_images = evolutionary_algorithm(source_image, **params)
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(fitness_history) + 1), fitness_history)
        plt.title(f"Fitness with Improvement: {improvement_name.capitalize()}")
        plt.xlabel("Generation")
        plt.ylabel("SSIM Fitness")
        plt.savefig(os.path.join(output_improvements_dir, f"fitness_full_{improvement_name}.png"))
        plt.close()
        plt.figure(figsize=(10, 5))
        plt.plot(range(1000, len(fitness_history) + 1), fitness_history[999:])
        plt.title(f"Fitness from Gen 1000 with Improvement: {improvement_name.capitalize()}")
        plt.xlabel("Generation")
        plt.ylabel("SSIM Fitness")
        plt.savefig(os.path.join(output_improvements_dir, f"fitness_1000_{improvement_name}.png"))
        plt.close()
        for i, img in enumerate(best_images):
            cv2.imwrite(os.path.join(output_improvements_dir, f"best_image_gen_{(i+1)*1000}_{improvement_name}.png"), img)

# Main execution
if __name__ == "__main__":
    try:
        source_image = cv2.imread('painting.png')
        if source_image is None:
            raise FileNotFoundError
    except FileNotFoundError:
        print("painting.png not found. Creating a dummy image.")
        source_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite('painting.png', source_image)
    run_experiments(source_image)
    print("Experiments complete. Check 'output' and 'output_improvements' directories for results.")