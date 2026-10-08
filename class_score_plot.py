import matplotlib.pyplot as plt

def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            if not line.startswith('#'): # If 'line' is not a header
                data.append([int(word) for word in line.split(',')])
    return data

if __name__ == '__main__':
    # Load score data
    class_kr = read_data('data/class_score_kr.csv')
    class_en = read_data('data/class_score_en.csv')

    # TODO) Prepare midterm, final, and total scores
    midterm_kr, final_kr = zip(*class_kr)
    total_kr = [40/125*midterm + 60/100*final for (midterm, final) in class_kr]
    midterm_en, final_en = zip(*class_en)
    total_en = [40/125 * midterm + 60/100 * final for (midterm, final) in class_en]

    # TODO) Plot midterm/final scores as points
    plt.subplot(2, 1, 1)
    plt.scatter(midterm_kr, final_kr,label = 'Korean', color = 'red', marker = '.')
    plt.scatter(midterm_en, final_en,label = 'English', color = 'blue', marker = '+')
    plt.xlabel("Midterm scores")
    plt.ylabel("Final scores")
    plt.axis([0, 125, 0, 100])
    plt.legend()
    plt.grid()

    # TODO) Plot total scores as a histogram

    plt.subplot(2, 1, 2)
    plt.hist(total_kr, bins = 20, range = (0, 100), color = 'red', alpha = 0.7, label = 'Korean')
    plt.hist(total_en, color = 'blue', bins = 20, range = (0, 100), alpha = 0.7, label = 'English')
    plt.legend()
    plt.axis([0, 100, 0, 9])
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
    plt.xlabel("Total scores")
    plt.ylabel("The number of students")

    plt.show()