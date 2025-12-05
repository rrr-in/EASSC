import pandas as pd
import numpy as np
import random
from typing import List, Tuple
import copy

class GeneticAlgorithmVRP:import pandas as pd
import numpy as np
import random
from typing import List, Tuple
import copy
import matplotlib.pyplot as plt
import folium
from folium import plugins

class GeneticAlgorithmVRP:
    def __init__(self, data_file, population_size=100, generations=500, 
                 crossover_rate=0.8, mutation_rate=0.2, vehicle_capacity=5000):
        # Load data
        self.df = pd.read_csv(data_file)
        self.vehicle_capacity = vehicle_capacity
        
        # Separate locations
        self.depot = self.df[self.df['type'] == 'depot'].iloc[0]
        self.parks = self.df[self.df['type'] == 'park'].to_dict('records')
        self.refills = self.df[self.df['type'] == 'refill'].to_dict('records')
        
        # GA parameters
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
        # Distance matrix
        self.distance_matrix = self._calculate_distance_matrix()
        
    def _calculate_distance_matrix(self):
        """Calculate Euclidean distance between all locations"""
        all_locations = [self.depot] + self.parks + self.refills
        n = len(all_locations)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    lat1, lon1 = all_locations[i]['lat'], all_locations[i]['lon']
                    lat2, lon2 = all_locations[j]['lat'], all_locations[j]['lon']
                    # Haversine distance (simplified as Euclidean for this case)
                    matrix[i][j] = np.sqrt((lat1-lat2)**2 + (lon1-lon2)**2)
        return matrix
    
    def get_distance(self, loc1_id, loc2_id):
        """Get distance between two locations by their IDs"""
        return self.distance_matrix[loc1_id][loc2_id]
    
    def create_initial_solution(self):
        """Create one valid initial solution"""
        route = []
        parks_to_visit = list(range(len(self.parks)))
        random.shuffle(parks_to_visit)
        
        current_load = self.vehicle_capacity
        
        for park_idx in parks_to_visit:
            park = self.parks[park_idx]
            demand = park['demand_liters']
            
            # If capacity insufficient, add refill
            if current_load < demand:
                # Add nearest refill station
                route.append(('refill', random.randint(0, len(self.refills)-1)))
                current_load = self.vehicle_capacity
            
            route.append(('park', park_idx))
            current_load -= demand
        
        return route
    
    def initialize_population(self):
        """Create initial population"""
        population = []
        for _ in range(self.population_size):
            solution = self.create_initial_solution()
            population.append(solution)
        return population
    
    def calculate_fitness(self, route):
        """Calculate fitness (lower is better)"""
        total_distance = 0
        total_time = 0
        current_load = self.vehicle_capacity
        current_pos = 0  # depot id
        
        penalty = 0
        
        for gene in route:
            gene_type, gene_idx = gene
            
            if gene_type == 'park':
                park = self.parks[gene_idx]
                park_id = park['id']
                demand = park['demand_liters']
                service_time = park['service_min']
                
                # Check capacity constraint
                if current_load < demand:
                    penalty += 10000  # Heavy penalty
                
                total_distance += self.get_distance(current_pos, park_id)
                total_time += service_time
                current_load -= demand
                current_pos = park_id
                
            elif gene_type == 'refill':
                refill = self.refills[gene_idx]
                refill_id = refill['id']
                service_time = refill['service_min']
                
                total_distance += self.get_distance(current_pos, refill_id)
                total_time += service_time
                current_load = self.vehicle_capacity  # Refill
                current_pos = refill_id
        
        # Return to depot
        total_distance += self.get_distance(current_pos, 0)
        
        # Fitness = weighted sum
        fitness = total_distance * 1000 + total_time + penalty
        return fitness
    
    def tournament_selection(self, population, fitnesses, tournament_size=5):
        """Tournament selection"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitnesses)]
        return population[winner_idx]
    
    def order_crossover(self, parent1, parent2):
        """Order Crossover (OX) for parks only"""
        # Extract only parks
        parks1 = [g for g in parent1 if g[0] == 'park']
        parks2 = [g for g in parent2 if g[0] == 'park']
        
        size = len(parks1)
        start, end = sorted(random.sample(range(size), 2))
        
        # Create child
        child_parks = [None] * size
        child_parks[start:end] = parks1[start:end]
        
        # Fill remaining from parent2
        pointer = 0
        for i in range(size):
            if child_parks[i] is None:
                while parks2[pointer] in child_parks:
                    pointer += 1
                child_parks[i] = parks2[pointer]
        
        # Insert refill stations
        child = self.insert_refill_stations(child_parks)
        return child
    
    def insert_refill_stations(self, parks_route):
        """Insert refill stations where needed"""
        route = []
        current_load = self.vehicle_capacity
        
        for park_gene in parks_route:
            park = self.parks[park_gene[1]]
            demand = park['demand_liters']
            
            if current_load < demand:
                # Insert refill
                refill_idx = random.randint(0, len(self.refills)-1)
                route.append(('refill', refill_idx))
                current_load = self.vehicle_capacity
            
            route.append(park_gene)
            current_load -= demand
        
        return route
    
    def mutate(self, route):
        """Mutation: swap two parks or remove/add refill"""
        mutated = copy.deepcopy(route)
        
        if random.random() < 0.5:
            # Swap two parks
            park_indices = [i for i, g in enumerate(mutated) if g[0] == 'park']
            if len(park_indices) >= 2:
                i, j = random.sample(park_indices, 2)
                mutated[i], mutated[j] = mutated[j], mutated[i]
        else:
            # Remove random refill or add new one
            refill_indices = [i for i, g in enumerate(mutated) if g[0] == 'refill']
            if refill_indices and random.random() < 0.5:
                mutated.pop(random.choice(refill_indices))
            else:
                # Add refill at random position
                pos = random.randint(0, len(mutated))
                refill_idx = random.randint(0, len(self.refills)-1)
                mutated.insert(pos, ('refill', refill_idx))
        
        return mutated
    
    def repair_solution(self, route):
        """Repair solution to ensure validity"""
        parks_only = [g for g in route if g[0] == 'park']
        return self.insert_refill_stations(parks_only)
    
    def run(self):
        """Run Genetic Algorithm"""
        # Initialize
        population = self.initialize_population()
        best_solution = None
        best_fitness = float('inf')
        fitness_history = []
        
        for generation in range(self.generations):
            # Calculate fitness
            fitnesses = [self.calculate_fitness(ind) for ind in population]
            
            # Track best
            gen_best_idx = np.argmin(fitnesses)
            if fitnesses[gen_best_idx] < best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_solution = copy.deepcopy(population[gen_best_idx])
            
            fitness_history.append(best_fitness)
            
            # Print progress
            if generation % 50 == 0:
                print(f"Generation {generation}: Best Fitness = {best_fitness:.2f}")
            
            # Create new population
            new_population = []
            
            # Elitism: keep best 10%
            elite_count = int(0.1 * self.population_size)
            elite_indices = np.argsort(fitnesses)[:elite_count]
            for idx in elite_indices:
                new_population.append(copy.deepcopy(population[idx]))
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child = self.order_crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(parent1)
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)
                
                # Repair
                child = self.repair_solution(child)
                
                new_population.append(child)
            
            population = new_population
        
        return best_solution, best_fitness, fitness_history
    
    def decode_solution(self, solution):
        """Decode solution to readable format"""
        route_details = []
        route_details.append({
            'step': 0,
            'type': 'DEPOT',
            'name': self.depot['name'],
            'lat': self.depot['lat'],
            'lon': self.depot['lon'],
            'demand': 0,
            'service_min': 0
        })
        
        for gene in solution:
            gene_type, gene_idx = gene
            if gene_type == 'park':
                park = self.parks[gene_idx]
                route_details.append({
                    'step': len(route_details),
                    'type': 'PARK',
                    'name': park['name'],
                    'demand': park['demand_liters'],
                    'service_min': park['service_min'],
                    'lat': park['lat'],
                    'lon': park['lon']
                })
            else:
                refill = self.refills[gene_idx]
                route_details.append({
                    'step': len(route_details),
                    'type': 'REFILL',
                    'name': refill['name'],
                    'demand': 0,
                    'service_min': refill['service_min'],
                    'lat': refill['lat'],
                    'lon': refill['lon']
                })
        
        route_details.append({
            'step': len(route_details),
            'type': 'DEPOT',
            'name': self.depot['name'],
            'lat': self.depot['lat'],
            'lon': self.depot['lon'],
            'demand': 0,
            'service_min': 0
        })
        
        return pd.DataFrame(route_details)
    
    def plot_convergence(self, fitness_history, save_path='convergence_plot.png'):
        """Plot GA convergence"""
        plt.figure(figsize=(12, 6))
        plt.plot(fitness_history, linewidth=2, color='#2E86AB')
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Best Fitness', fontsize=12)
        plt.title('Genetic Algorithm Convergence', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved: {save_path}")
        plt.close()
    
    def visualize_route_map(self, route_df, save_path='route_map.html'):
        """Create interactive map with Folium"""
        # Center map on depot
        center_lat = self.depot['lat']
        center_lon = self.depot['lon']
        
        # Create map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Add markers
        for idx, row in route_df.iterrows():
            lat, lon = row['lat'], row['lon']
            
            if row['type'] == 'DEPOT':
                folium.Marker(
                    location=[lat, lon],
                    popup=f"<b>{row['name']}</b><br>DEPOT",
                    icon=folium.Icon(color='red', icon='home', prefix='fa'),
                    tooltip='DEPOT'
                ).add_to(m)
            elif row['type'] == 'PARK':
                folium.Marker(
                    location=[lat, lon],
                    popup=f"<b>{row['name']}</b><br>Demand: {row['demand']} L<br>Service: {row['service_min']} min",
                    icon=folium.Icon(color='green', icon='tree', prefix='fa'),
                    tooltip=f"Park #{row['step']}"
                ).add_to(m)
            elif row['type'] == 'REFILL':
                folium.Marker(
                    location=[lat, lon],
                    popup=f"<b>{row['name']}</b><br>REFILL STATION<br>Service: {row['service_min']} min",
                    icon=folium.Icon(color='blue', icon='tint', prefix='fa'),
                    tooltip='Refill Station'
                ).add_to(m)
        
        # Draw route lines
        route_coords = [[row['lat'], row['lon']] for _, row in route_df.iterrows()]
        folium.PolyLine(
            route_coords,
            color='#FF6B35',
            weight=3,
            opacity=0.8,
            popup='Route'
        ).add_to(m)
        
        # Add route arrows
        plugins.AntPath(
            route_coords,
            color='#FF6B35',
            weight=3,
            opacity=0.6,
            delay=1000
        ).add_to(m)
        
        # Save map
        m.save(save_path)
        print(f"Interactive map saved: {save_path}")
        return m
    
    def plot_route_static(self, route_df, save_path='route_static.png'):
        """Create static route visualization"""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot all locations
        depot_data = route_df[route_df['type'] == 'DEPOT'].iloc[0]
        parks_data = route_df[route_df['type'] == 'PARK']
        refills_data = route_df[route_df['type'] == 'REFILL']
        
        # Plot parks
        ax.scatter(parks_data['lon'], parks_data['lat'], 
                  c='green', s=100, marker='o', label='Parks', 
                  alpha=0.7, edgecolors='black', linewidth=1.5)
        
        # Plot refills
        ax.scatter(refills_data['lon'], refills_data['lat'], 
                  c='blue', s=150, marker='s', label='Refill Stations',
                  alpha=0.7, edgecolors='black', linewidth=1.5)
        
        # Plot depot
        ax.scatter(depot_data['lon'], depot_data['lat'], 
                  c='red', s=300, marker='*', label='Depot',
                  edgecolors='black', linewidth=2)
        
        # Draw route
        route_coords = list(zip(route_df['lon'], route_df['lat']))
        for i in range(len(route_coords) - 1):
            x1, y1 = route_coords[i]
            x2, y2 = route_coords[i + 1]
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=1.5, 
                                     color='orange', alpha=0.6))
        
        # Add step numbers for parks
        for idx, row in parks_data.iterrows():
            ax.text(row['lon'], row['lat'], str(row['step']), 
                   fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title('Optimized Water Truck Route', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Static route plot saved: {save_path}")
        plt.close()
    
    def create_summary_statistics(self, route_df):
        """Create summary statistics"""
        total_parks = len(route_df[route_df['type'] == 'PARK'])
        total_refills = len(route_df[route_df['type'] == 'REFILL'])
        total_demand = route_df[route_df['type'] == 'PARK']['demand'].sum()
        total_service_time = route_df['service_min'].sum()
        
        # Calculate total distance
        total_distance = 0
        for i in range(len(route_df) - 1):
            lat1, lon1 = route_df.iloc[i]['lat'], route_df.iloc[i]['lon']
            lat2, lon2 = route_df.iloc[i+1]['lat'], route_df.iloc[i+1]['lon']
            total_distance += np.sqrt((lat1-lat2)**2 + (lon1-lon2)**2)
        
        summary = {
            'Total Parks Visited': total_parks,
            'Total Refill Stops': total_refills,
            'Total Water Demand (L)': total_demand,
            'Total Service Time (min)': total_service_time,
            'Estimated Distance': f"{total_distance:.4f}",
            'Average Demand per Park (L)': total_demand / total_parks if total_parks > 0 else 0
        }
        
        return summary


# ===== CARA PENGGUNAAN =====
if __name__ == "__main__":
    # Jalankan GA
    ga = GeneticAlgorithmVRP(
        data_file='your_data.csv',  # Ganti dengan file CSV Anda
        population_size=100,
        generations=500,
        crossover_rate=0.8,
        mutation_rate=0.2,
        vehicle_capacity=5000
    )
    
    print("Menjalankan Genetic Algorithm...")
    best_route, best_fitness, history = ga.run()
    
    print(f"\n{'='*50}")
    print(f"HASIL OPTIMASI")
    print(f"{'='*50}")
    print(f"Best Fitness: {best_fitness:.2f}")
    
    # Decode rute
    route_df = ga.decode_solution(best_route)
    
    # Summary statistics
    summary = ga.create_summary_statistics(route_df)
    print("\n=== SUMMARY STATISTICS ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Visualisasi
    print(f"\n{'='*50}")
    print("MEMBUAT VISUALISASI...")
    print(f"{'='*50}")
    
    # 1. Plot convergence
    ga.plot_convergence(history)
    
    # 2. Interactive map
    ga.visualize_route_map(route_df)
    
    # 3. Static plot
    ga.plot_route_static(route_df)
    
    # Simpan rute detail
    route_df.to_csv('optimal_route.csv', index=False)
    print("\nRoute details saved: optimal_route.csv")
    
    print(f"\n{'='*50}")
    print("SELESAI!")
    print(f"{'='*50}")
    print("\nFile yang dihasilkan:")
    print("1. optimal_route.csv - Detail rute lengkap")
    print("2. convergence_plot.png - Grafik konvergensi GA")
    print("3. route_map.html - Peta interaktif (buka di browser)")
    print("4. route_static.png - Visualisasi rute statis")
    def __init__(self, data_file, population_size=100, generations=500, 
                 crossover_rate=0.8, mutation_rate=0.2, vehicle_capacity=5000):
        # Load data
        self.df = pd.read_csv(r"C:\Users\punya karina\Downloads\SC\FinalProject\nodes.csv")
        self.vehicle_capacity = vehicle_capacity
        
        # Separate locations
        self.depot = self.df[self.df['type'] == 'depot'].iloc[0]
        self.parks = self.df[self.df['type'] == 'park'].to_dict('records')
        self.refills = self.df[self.df['type'] == 'refill'].to_dict('records')
        
        # GA parameters
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
        # Distance matrix
        self.distance_matrix = self._calculate_distance_matrix()
        
    def _calculate_distance_matrix(self):
        """Calculate Euclidean distance between all locations"""
        all_locations = [self.depot] + self.parks + self.refills
        n = len(all_locations)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    lat1, lon1 = all_locations[i]['lat'], all_locations[i]['lon']
                    lat2, lon2 = all_locations[j]['lat'], all_locations[j]['lon']
                    # Haversine distance (simplified as Euclidean for this case)
                    matrix[i][j] = np.sqrt((lat1-lat2)**2 + (lon1-lon2)**2)
        return matrix
    
    def get_distance(self, loc1_id, loc2_id):
        """Get distance between two locations by their IDs"""
        return self.distance_matrix[loc1_id][loc2_id]
    
    def create_initial_solution(self):
        """Create one valid initial solution"""
        route = []
        parks_to_visit = list(range(len(self.parks)))
        random.shuffle(parks_to_visit)
        
        current_load = self.vehicle_capacity
        
        for park_idx in parks_to_visit:
            park = self.parks[park_idx]
            demand = park['demand_liters']
            
            # If capacity insufficient, add refill
            if current_load < demand:
                # Add nearest refill station
                route.append(('refill', random.randint(0, len(self.refills)-1)))
                current_load = self.vehicle_capacity
            
            route.append(('park', park_idx))
            current_load -= demand
        
        return route
    
    def initialize_population(self):
        """Create initial population"""
        population = []
        for _ in range(self.population_size):
            solution = self.create_initial_solution()
            population.append(solution)
        return population
    
    def calculate_fitness(self, route):
        """Calculate fitness (lower is better)"""
        total_distance = 0
        total_time = 0
        current_load = self.vehicle_capacity
        current_pos = 0  # depot id
        
        penalty = 0
        
        for gene in route:
            gene_type, gene_idx = gene
            
            if gene_type == 'park':
                park = self.parks[gene_idx]
                park_id = park['id']
                demand = park['demand_liters']
                service_time = park['service_min']
                
                # Check capacity constraint
                if current_load < demand:
                    penalty += 10000  # Heavy penalty
                
                total_distance += self.get_distance(current_pos, park_id)
                total_time += service_time
                current_load -= demand
                current_pos = park_id
                
            elif gene_type == 'refill':
                refill = self.refills[gene_idx]
                refill_id = refill['id']
                service_time = refill['service_min']
                
                total_distance += self.get_distance(current_pos, refill_id)
                total_time += service_time
                current_load = self.vehicle_capacity  # Refill
                current_pos = refill_id
        
        # Return to depot
        total_distance += self.get_distance(current_pos, 0)
        
        # Fitness = weighted sum
        fitness = total_distance * 1000 + total_time + penalty
        return fitness
    
    def tournament_selection(self, population, fitnesses, tournament_size=5):
        """Tournament selection"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitnesses)]
        return population[winner_idx]
    
    def order_crossover(self, parent1, parent2):
        """Order Crossover (OX) for parks only"""
        # Extract only parks
        parks1 = [g for g in parent1 if g[0] == 'park']
        parks2 = [g for g in parent2 if g[0] == 'park']
        
        size = len(parks1)
        start, end = sorted(random.sample(range(size), 2))
        
        # Create child
        child_parks = [None] * size
        child_parks[start:end] = parks1[start:end]
        
        # Fill remaining from parent2
        pointer = 0
        for i in range(size):
            if child_parks[i] is None:
                while parks2[pointer] in child_parks:
                    pointer += 1
                child_parks[i] = parks2[pointer]
        
        # Insert refill stations
        child = self.insert_refill_stations(child_parks)
        return child
    
    def insert_refill_stations(self, parks_route):
        """Insert refill stations where needed"""
        route = []
        current_load = self.vehicle_capacity
        
        for park_gene in parks_route:
            park = self.parks[park_gene[1]]
            demand = park['demand_liters']
            
            if current_load < demand:
                # Insert refill
                refill_idx = random.randint(0, len(self.refills)-1)
                route.append(('refill', refill_idx))
                current_load = self.vehicle_capacity
            
            route.append(park_gene)
            current_load -= demand
        
        return route
    
    def mutate(self, route):
        """Mutation: swap two parks or remove/add refill"""
        mutated = copy.deepcopy(route)
        
        if random.random() < 0.5:
            # Swap two parks
            park_indices = [i for i, g in enumerate(mutated) if g[0] == 'park']
            if len(park_indices) >= 2:
                i, j = random.sample(park_indices, 2)
                mutated[i], mutated[j] = mutated[j], mutated[i]
        else:
            # Remove random refill or add new one
            refill_indices = [i for i, g in enumerate(mutated) if g[0] == 'refill']
            if refill_indices and random.random() < 0.5:
                mutated.pop(random.choice(refill_indices))
            else:
                # Add refill at random position
                pos = random.randint(0, len(mutated))
                refill_idx = random.randint(0, len(self.refills)-1)
                mutated.insert(pos, ('refill', refill_idx))
        
        return mutated
    
    def repair_solution(self, route):
        """Repair solution to ensure validity"""
        parks_only = [g for g in route if g[0] == 'park']
        return self.insert_refill_stations(parks_only)
    
    def run(self):
        """Run Genetic Algorithm"""
        # Initialize
        population = self.initialize_population()
        best_solution = None
        best_fitness = float('inf')
        fitness_history = []
        
        for generation in range(self.generations):
            # Calculate fitness
            fitnesses = [self.calculate_fitness(ind) for ind in population]
            
            # Track best
            gen_best_idx = np.argmin(fitnesses)
            if fitnesses[gen_best_idx] < best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_solution = copy.deepcopy(population[gen_best_idx])
            
            fitness_history.append(best_fitness)
            
            # Print progress
            if generation % 50 == 0:
                print(f"Generation {generation}: Best Fitness = {best_fitness:.2f}")
            
            # Create new population
            new_population = []
            
            # Elitism: keep best 10%
            elite_count = int(0.1 * self.population_size)
            elite_indices = np.argsort(fitnesses)[:elite_count]
            for idx in elite_indices:
                new_population.append(copy.deepcopy(population[idx]))
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child = self.order_crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(parent1)
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)
                
                # Repair
                child = self.repair_solution(child)
                
                new_population.append(child)
            
            population = new_population
        
        return best_solution, best_fitness, fitness_history
    
    def decode_solution(self, solution):
        """Decode solution to readable format"""
        route_details = []
        route_details.append({
            'step': 0,
            'type': 'DEPOT',
            'name': self.depot['name'],
            'lat': self.depot['lat'],
            'lon': self.depot['lon']
        })
        
        for gene in solution:
            gene_type, gene_idx = gene
            if gene_type == 'park':
                park = self.parks[gene_idx]
                route_details.append({
                    'step': len(route_details),
                    'type': 'PARK',
                    'name': park['name'],
                    'demand': park['demand_liters'],
                    'service_min': park['service_min'],
                    'lat': park['lat'],
                    'lon': park['lon']
                })
            else:
                refill = self.refills[gene_idx]
                route_details.append({
                    'step': len(route_details),
                    'type': 'REFILL',
                    'name': refill['name'],
                    'service_min': refill['service_min'],
                    'lat': refill['lat'],
                    'lon': refill['lon']
                })
        
        route_details.append({
            'step': len(route_details),
            'type': 'DEPOT',
            'name': self.depot['name'],
            'lat': self.depot['lat'],
            'lon': self.depot['lon']
        })
        
        return pd.DataFrame(route_details)


# ===== CARA PENGGUNAAN =====
if __name__ == "__main__":
    # Jalankan GA
    ga = GeneticAlgorithmVRP(
        data_file='your_data.csv',  # Ganti dengan file CSV Anda
        population_size=100,
        generations=500,
        crossover_rate=0.8,
        mutation_rate=0.2,
        vehicle_capacity=5000
    )
    
    print("Menjalankan Genetic Algorithm...")
    best_route, best_fitness, history = ga.run()
    
    print(f"\n=== HASIL OPTIMASI ===")
    print(f"Best Fitness: {best_fitness:.2f}")
    print(f"\nJumlah Park dikunjungi: {sum(1 for g in best_route if g[0] == 'park')}")
    print(f"Jumlah Refill digunakan: {sum(1 for g in best_route if g[0] == 'refill')}")
    
    # Decode dan tampilkan rute
    route_df = ga.decode_solution(best_route)
    print("\n=== RUTE DETAIL ===")
    print(route_df.to_string(index=False))
    
    # Simpan hasil
    route_df.to_csv('optimal_route.csv', index=False)