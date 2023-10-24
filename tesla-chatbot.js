import { CheerioCrawler, Dataset } from 'crawlee';
import { Actor } from 'apify';
import { Readability } from '@mozilla/readability';
import { JSDOM } from 'jsdom';
import TurndownService from 'turndown';
import { CSVLoader } from "langchain/document_loaders/fs/csv";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RetrievalQAChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models/openai";

const URL = 'https://www.tesla.com/ownersmanual/model3/en_us/';
const QUESTION = 'Explain what autopilot is and provide step by step instructions to turning it on.';

const turndownService = new TurndownService();

// If you're getting 403 errors, you might have to use proxies.
// This is the simplest way to add them to the crawler, but it requires
// an Apify account with proxies enabled.

// const proxyConfiguration = await Actor.createProxyConfiguration({
//     countryCode: 'US',
// })

const crawler = new CheerioCrawler({
    // proxyConfiguration,
    requestHandler: async ({ body, request, enqueueLinks }) => {
        console.log('Parsing', request.url)
        const dom = new JSDOM(body, { url: request.loadedUrl });
        const reader = new Readability(dom.window.document, {
            charThreshold: 200,
        });

        const { content } = reader.parse();
        const markdown = turndownService.turndown(content);

        await Dataset.pushData({
            request,
            markdown,
        })
        await enqueueLinks({
            globs: [`${URL}/**`]
        })
    }
});

await crawler.run([URL])

console.log('Exporting crawled results to CSV.');
await Dataset.exportToCSV('tesla-docs');

const loader = new CSVLoader(
    "storage/key_value_stores/default/tesla-docs.csv",
    "markdown",
);

console.log('Loading results to LangChain.');
const data = await loader.load();

const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
});

console.log('Splitting markdown pages into LangChain documents.');
const splitDocs = await textSplitter.splitDocuments(data);

const embeddings = new OpenAIEmbeddings();

console.log('Creating a vector database.');
const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
);

const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0,
});
const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

console.log('Asking ChatGPT to answer the question:', QUESTION);
const response = await chain.call({
    query: QUESTION,
});
console.log('Response:', response.text);
