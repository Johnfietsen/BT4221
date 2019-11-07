import pandas as pd
import argparse


def main():
	"""
	Main function
	"""

	profit = dict()
	ratio = pd.read_csv(flags.model + '_ratio.tsv', sep='\t')
	ratio = ratio.merge(pd.read_csv('../data/stocks_data/change.csv'))

	for file in ratio['File']:
		tmp_row = ratio.loc[ratio['File'] == file]
		if tmp_row['ratio neg/pos'].iloc[0] > flags.threshold:
		   	profit[file] = flags.investment * tmp_row[flags.hour].iloc[0]
		elif tmp_row['ratio neg/pos'].iloc[0] < flags.threshold:
		   	profit[file] = - flags.investment * tmp_row[flags.hour].iloc[0]

	total = sum([profit[file] for file in profit])
	total_normalized = total / len(profit)

	f = open(flags.model + '_naive_profits.csv', 'w')
	for file in profit:
	    print(file, profit[file])
	    f.write(file + ',' + str(profit[file]) + '\n')
	print('total', total)
	print('norm', total_normalized)
	f.write('total' + ',' + str(total) + '\n')
	f.write('norm' + ',' + str(total_normalized) + '\n')



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type = str, default = 'both',
						help='both, sanders, or semeval')
	parser.add_argument('--threshold', type = int, default = 1,
						help='ratio at which trading decision is made')
	parser.add_argument('--investment', type = float, default = 100,
						help='how much money is put in each investment')
	parser.add_argument('--hour', type = str, default = str(15),
						help='time at which stock is sold')
	flags, unparsed = parser.parse_known_args()

	main()
