import json
import os

with open('latex_tab.txt', 'w') as outfile:
    outfile.write(f'\\begin{{figure}}[htb]\n  \\centering\n')
    for i in range(25):
        outfile.write(f'\\begin{{minipage}}{{0.4\textwidth}}\n \\centering\n \\includegraphics[width=0.4\
\\textwidth]\\includegraphics[scale=0.45]{{unbtex-example/figs/hre/instantaneous_velocity_hre1.png}}\n\
\\caption{{Gráfico Velocidades Instantâneas hre1}}\\label{{inst-hre1}}
\\legend{Fonte: Produzido pelo autor}
  \end{minipage}
  \hfill')
    
if __name__ == '__main__':
    main()