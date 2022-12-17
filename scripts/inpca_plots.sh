
tasks=("[0,1,2,3,4]" "[95,96,97,98,99]" "[6,11,16,21,26]" "[56,58,62,66,68]")


for task in "${tasks[@]}"
do
    python inpca_plots.py +plot_inpca=default plot_inpca.task_labs=$task
done