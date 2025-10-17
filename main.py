import urllib.request, os
from src.preprocessing.data_preprocessing import data_preprocessing
from src.models.genetic_algorithm import run_genetic_algorithm
from src.models.mixed_integer_programmin import run_cvrp_mip

def main():

    data_preprocessing()

    run_genetic_algorithm()

    run_cvrp_mip()
    



if __name__ == "__main__":
    main()