import random
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Wczytanie zbioru danych Iris
iris = load_iris()
X = iris.data
y = iris.target

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)

# Funkcja oceny - dokładność klasyfikacji
def evaluate(individual, X, y):
    C = individual[0]
    gamma = individual[1]
    model = SVC(C=C, gamma=gamma)
    model.fit(X, y)
    y_pred = model.predict(X)
    return accuracy_score(y, y_pred)

# Inicjalizacja populacji początkowej
def inicjalizacja_populacji(rozmiar_populacji):
    population = []
    for i in range(rozmiar_populacji):
        individual = [random.uniform(0.1, 10), random.uniform(0.001, 1)]
        population.append(individual)
    return population

# Selekcja osobników na podstawie strategii turniejowej
def selection(population, scores, k):
    selected = []
    while len(selected) < k:
        sorted_population = [x for i, x in sorted(zip(scores, population))]
        selected.append(random.choice(sorted_population))
    return selected


# Krzyżowanie osobników
def crossover(parent1, parent2):
    child = [(parent1[i] + parent2[i]) / 2 for i in range(len(parent1))]
    return child

# Mutacja osobnika
def mutation(individual, mutowanie):
    mutated_individual = individual.copy()
    for i in range(len(mutated_individual)):
        if random.random() < mutowanie:
            mutated_individual[i] = random.uniform(0.1, 10) if i == 0 else random.uniform(0.001, 1)
    return mutated_individual

# Algorytm genetyczny
def algorytm_genetyczny(rozmiar_populacji, ilosc_generacji, rozmiar_turnieju, mutowanie, X, y):
    population = inicjalizacja_populacji(rozmiar_populacji)
    best_individual = None
    best_score = 0

    for generation in range(ilosc_generacji):
        scores = [evaluate(individual, X_train, y_train) for individual in population]

        for i, score in enumerate(scores):
            if score > best_score:

                best_individual = population[i]
                best_score = score

        selected = selection(population, scores, rozmiar_turnieju)

        offspring = []
        for i in range(0, min(rozmiar_populacji-1, len(selected)-1), 2):
            parent1 = selected[i]
            parent2 = random.choice(selected[i+1:] + selected[:i])
            child = crossover(parent1, parent2)
            mutated_child = mutation(child, mutowanie)
            offspring.append(mutated_child)

        population = offspring

        #Zmiana ostatniego osobnika w populacji najlepszym osobnikiem
        last_individual = population[-1]
        last_score = evaluate(last_individual, X_train, y_train)
        if last_score < best_score:
            population[-1] = best_individual

    return best_individual

# Wywołanie algorytmu genetycznego
rozmiar_populacji = 50
ilosc_generacji = 200
rozmiar_turnieju = 10
mutowanie = 0.2
best_individual = algorytm_genetyczny(rozmiar_populacji, ilosc_generacji, rozmiar_turnieju, mutowanie, X_train, y_train)

# Ewaluacja najlepszego osobnika na zbiorze testowym
best_model = SVC(C=best_individual[0], gamma=best_individual[1])
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Dokładność modelu:", accuracy)
print("Najlepszy osobnik", best_individual)