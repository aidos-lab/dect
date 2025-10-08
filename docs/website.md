# Website 


This website is build using the `mkdocs` package. To build the project locally for 
testing, run `uv run mkdocs serve`. 

All files for the website are located in the `docs/` folder and can be edited there as 
markdown files. The website is build and published using GitHub actions. 

If you intend to add a new page to the website, please add a new markdown file under the 
docs folder and edit the `nav` section in the `mkdocs.yml` file to include it in the 
navigation bar. 

# Submitting Notebooks 

The notebooks in the `notebooks` can be converted _locally_ to to markdown files with 
the `make notebooks` command. Under the hood it uses nbconvert to run the notebooks and 
convert them to `.md` files. As this may depend on hardware acceleration the command has 
to be ran locally before commiting to the repository. The outputs are stored under the 
`docs/notebooks` folder and new notebooks have to be added manually to the navigation bar
in the `mkdocs.yml` file. 

