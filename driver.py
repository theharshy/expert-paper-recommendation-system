from ExpertSystem import Paper
from ExpertSystem import Expert
from ExpertSystem import ExpertSystem
import io

# Function that Return an Array of all Expert with their attributed Paper array
def get_expert_papers(expert_list_file, expert_dir):
    reviewers_dict = {}

    author_format = expert_dir + "/{}.txt"
    with open(expert_list_file, 'r') as file:
        expert_names = file.readlines()

        for author in expert_names:
            # creating expert objects and adding then to dictionary
            author = (author.strip())
            with io.open(author_format.format(author), 'r',
                         encoding='utf-8', errors='ignore') as infile, \
                io.open('temp.txt', 'w', encoding='utf-8', errors='ignore') \
                    as outfile:
                for line in infile:
                    print(*line.split(), file=outfile)

            file1 = open("temp.txt", "r")
            for eachline in file1.readlines():
                p1 = Paper(eachline, author, -1, True)
                reviewer = reviewers_dict.get(author)
                if(reviewer is not None):
                    reviewer.papers.append(p1)
                else:
                    e1 = Expert(p1, author)
                    reviewers_dict[author] = e1

    print(len(reviewers_dict))
    return reviewers_dict


# Returns an array of Papers for which Experts are to be allocated
def get_paper_candidates(title_file, abstract_file):
        research_papers = []

        # now for papers for which we neeed recommendation
        file = open(title_file, "r")
        lines = file.readlines()

        with io.open(abstract_file, 'r', encoding='utf-8',
                     errors='ignore') as infile, \
                io.open('temp.txt', 'w', encoding='utf-8', errors='ignore') \
                as outfile:

            for line in infile:
                print(*line.split(), file=outfile)

        file1 = open("temp.txt", "r")
        contents = file1.readlines()
        # list of paper objects for which we need recommendation
        research_papers = []

        # adding to research paper list
        for line, content in zip(lines, contents):
            line = line.strip()
            content = str(line) + str(content)
            p1 = Paper(content, -1, -1, False)
            research_papers.append(p1)

        print(len(research_papers))
        return research_papers


# assigning unique index to Each paper
def assign_paper_uid(expertlist, research_papers):
    i = 0
    # assigning unique index to each paper
    all_paper = []
    all_paper_objects = []
    for expert in expertlist:
        for paper in expert.papers:
            paper.assignindex(i)
            all_paper.append(paper.contents)
            all_paper_objects.append(paper)
            i = i + 1

    for paper in research_papers:
        paper.assignindex(i)
        all_paper.append(paper.contents)
        all_paper_objects.append(paper)
        i = i + 1

    return (all_paper, all_paper_objects)


# Outputs the results to the file
def save_to_file(output_file, recommend):
    with open(output_file, 'w') as output:
        for paper in recommend:
            for expert, score in paper:
                output.write("{}:{},".format(expert, score))

            output.write("\n")


def main():
    method = 2
    output_file = 'output.txt'

    # a dictionary of Expert object indexed by names
    print("Loading Expert Data..")
    reviewers_dict = get_expert_papers('evaluation/short_listed_revs.txt',
                                       'title_abstract_data')
    print("Loading Paper Candidates")
    research_papers = get_paper_candidates("evaluation/titles.txt",
                                           'evaluation/abstracts.txt')
    expertlist = list(reviewers_dict.values())
    (all_paper, all_paper_objects) = assign_paper_uid(expertlist,
                                                      research_papers)
    print("Generating the TF-IDF Matrix..")
    ES = ExpertSystem(n_clusters=200, experts_per_paper=15)
    tfidf_mat = ES.get_tfidf(all_paper)

    # generating similarity matrix
    print("Generating the similarity matrix..")
    similarity = ES.get_cosine_sim(tfidf_mat)

    labels = ES.get_kmeans(tfidf_mat)

    print("Computing Relavant Experts for Candidate Papers")
    recommend = []
    if method is 1:
        recommend = ES.algorithm1(similarity, expertlist, research_papers,
                                  all_paper_objects)
    elif method is 2:
        recommend = ES.algorithm2(labels, similarity, expertlist,
                                  research_papers)

    print('Saving Results to file...')
    save_to_file(output_file, recommend)

    print("Done!")


if __name__ == '__main__':
        main()
