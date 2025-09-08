1. Download the zip file(Askracine) from GDrive and unzip it.
2. Install Ollama in your system for Llama3.2
3. Create an environment in Visual Studio Code(Install all the libraries in it)
4. Install all the libraries mentioned in the requirements.txt file
(if any library is missing here you can install it using pip)
5. Install React
1. Install node manager from their
website(https://nodejs.org/dist/v22.15.1/node-v22.15.1-x64.msi)
2.then enter this command in terminal: npx create-react-app ‘app name’
6. Copy the files from “UI” in Askracine folder as it is and replace it in your
‘app name’
Execution:
Backend:
1. Activate the environment
2. Run database.py (Run once to create the database)
3. Run preprocess.py (Run initially for first time and run whenever the
company data is updated)
4. Run scrapper.py (Run initially for first time and run whenever the
website is updated)
5. Run indexer.py (Run initially for first time and run when any of either 3
or 4 is updated)
6. Run one.py the backend always to work with chatbot
[NOTE: for the preprocess.py code we were not having proper formatted company
data so we tried to give this as a provision for further development feed company data
into this code in some format and execute it to continue other operations, this data is
stored in company_knowledge_base.db]
Frontend:
1. Locate to react app of yours using ‘cd’ to app_name
2. Npm start(frontend up and running)
Contents:
1. Requirements.txt: has all the required libraries to be installed
2. ui: react frontend code of the project
3. Faiss_index: stores all vector embeddings in it(created when codes are
executed)
4. Python files: 5 python codes contain the required background code.
5. company_knowledge_base: database that stores company’s
data(created when codes are executed)
6. scrapped_data.db: database that stores web-scrapped content(created
when codes are executed)
