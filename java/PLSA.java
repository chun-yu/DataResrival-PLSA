import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.TreeMap;
import java.util.Vector;
import java.lang.*;
import java.math.BigDecimal;


public class PLSA{
	public static Map<String, String> indexMap = new HashMap<String, String>();
	public static Map<String, Integer> indexIDFMap = new HashMap<String, Integer>(); //index idf to doc
	public static Map<String, String> docMap = new HashMap<String, String>();
	public static Map<String, HashMap<String, Integer>> tfMap = new HashMap<String, HashMap<String, Integer>>();
	public static Map<String, HashMap<String, Integer>> DoctfMap = new HashMap<String, HashMap<String, Integer>>(); //doc word tf
	public static Map<String, Integer> docWordNumMap = new HashMap<String, Integer>(); //doc word count map
	public static Map<String, Double> resultMap = new HashMap<String, Double>();

	StringBuilder report = new StringBuilder();
	
	public void storeQuery(String filename){
		String line;
		StringBuilder sValue = new StringBuilder("");
			try {
				FileReader fr=new FileReader(new File("./Query/"+filename));
				BufferedReader br=new BufferedReader(fr);
				while ((line=br.readLine()) != null) {
					line = line.substring(0,line.indexOf('-'));
					sValue.append(line);
				}
				indexMap.put(filename, sValue.toString());
				//storeDoc(sValue.toString(),"doc_list.txt");
				br.close();
			}
			catch (IOException e) {System.out.println(e);}
		indexMap = sortHashMapByComparator(indexMap);
		setTerms();
	}
	public void setTerms(){
		String docword;
		int maxFreq = 0;
		for (Map.Entry<String, String> entry : indexMap.entrySet()){
		    String term = entry.getValue().toString();
			String[] terms = term.split(" ");
			HashMap<String, Integer> freqMap = new HashMap<String, Integer>();
			for(String s: terms){
				if(freqMap.containsKey(s)){
					int currentFreq = freqMap.get(s);
					freqMap.put(s, currentFreq+1);
					if(currentFreq+1 > maxFreq) maxFreq = currentFreq+1;
				}else{
					freqMap.put(s, 1);
				}
				freqMap.put("MAX", maxFreq);
				if(indexIDFMap.containsKey(s)){
				}else{
					for(Map.Entry<String, HashMap<String,Integer>> entryDoc : DoctfMap.entrySet()){
						HashMap<String, Integer> freqMap2=entryDoc.getValue();
						if(freqMap2.containsKey(s)){
							if(indexIDFMap.containsKey(s)){
								if(indexIDFMap.get(s)<2265){
									int currentFreq = indexIDFMap.get(s);
									indexIDFMap.put(s, currentFreq+1);
								}else{}
							}else{
									indexIDFMap.put(s, 1);
							}
						}
					}
					if(!(indexIDFMap.containsKey(s)))	indexIDFMap.put(s,0);
				}
			}
			tfMap.put(entry.getKey().toString(),freqMap);
		}
	}
	public void storeDoc(String filename){
		StringBuilder sValue = new StringBuilder("");
		String sCurrentLine;
			try {
				BufferedReader br=new BufferedReader(new FileReader(new File("./Document/"+filename)));
				sCurrentLine=br.readLine();sCurrentLine=br.readLine();sCurrentLine=br.readLine();
				while ((sCurrentLine = br.readLine()) != null) {
					sCurrentLine = sCurrentLine.substring(0,sCurrentLine.indexOf('-'));
					sValue.append(sCurrentLine);
				}
				docMap.put(filename, sValue.toString());
				br.close();
			}
			catch (IOException e) {System.out.println(e);}
		docMap = sortHashMapByComparator(docMap);
	}
	public void setDocTF(){
		HashMap<String, Integer> freqMap;
		String name;String[] docterms;
		for (Map.Entry<String, String> entry : docMap.entrySet()){
			freqMap = new HashMap<String, Integer>();
			name = entry.getKey().toString();
			docterms=entry.getValue().toString().split(" ");
			for(String s: docterms){
				if(freqMap.containsKey(s)){
					int currentFreq = freqMap.get(s);
					freqMap.put(s, currentFreq+1);
				}else{
					freqMap.put(s, 1);
				}
				if(docWordNumMap.containsKey(s)){
					int currentFreq = docWordNumMap.get(s);
					docWordNumMap.put(s, currentFreq+1);
				}else{
					docWordNumMap.put(s,1);
				}
			}
			DoctfMap.put(name, freqMap);
		}
		System.out.println(docWordNumMap.size()+"//");
	}


	public void retrieve(){
		PLSAEM plsa = new PLSAEM();
		plsa.doplsa(indexMap,indexIDFMap,docMap,tfMap,DoctfMap,docWordNumMap);
		Map<String, Integer> queryDocMap = new HashMap<String, Integer>();
		String[] terms;String queryNmae;
		report.append("Query,RetrievedDocuments"+"\n");
		for (Map.Entry<String, String> Indexentry : indexMap.entrySet()){
			queryNmae=Indexentry.getKey().toString();
			resultMap = plsa.getPlsa5(queryNmae);

			resultMap = sortMapResult(resultMap);
			report.append(Indexentry.getKey()+",");
			for (Map.Entry<String, Double> resultentry : resultMap.entrySet()){
				//System.out.println(resultentry.getKey()+" // "+resultentry.getValue());
				report.append(resultentry.getKey()+" ");
			}
			report.append("\n");
			System.out.println(queryNmae+"  FINISH  !!");
		}
		writeTestResults("submission.txt");
	}

	public HashMap<String, String> sortHashMapByComparator(Map<String, String> unsortMap) {
		// Convert Map to List
		List<Map.Entry <String, String>> list = 
			new LinkedList<Map.Entry<String, String>>(unsortMap.entrySet());
		// Sort list with comparator, to compare the Map values
		Collections.sort(list, new Comparator<Map.Entry<String, String>>() {
			public int compare(Map.Entry<String, String> o1,
					Map.Entry<String, String> o2) {
				return o1.getKey().compareTo(o2.getKey());
			}
		});
		// Convert sorted map back to a Map
		HashMap<String, String> sortedMap = new LinkedHashMap<String, String>();
		for (Iterator<Map.Entry<String, String>> it = list.iterator(); it.hasNext();) {
			Map.Entry<String, String> entry = it.next();
			sortedMap.put(entry.getKey(), entry.getValue());
		}
		return sortedMap;
	}

	public void writeTestResults(String filename){
		Writer writer = null;
		try {
			writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "utf-8"));
			writer.write(report.toString());
		} catch (IOException ex) {
			ex.printStackTrace();
		} finally {
		   try {writer.close();} catch (Exception ex) {/*ignore*/}
		}
	}
	public HashMap<String, Double> sortMapResult(Map<String, Double> unsortMap) {
		// Convert Map to List
		List<Map.Entry <String, Double>> list = 
			new LinkedList<Map.Entry<String, Double>>(unsortMap.entrySet());
		// Sort list with comparator, to compare the Map values
		Collections.sort(list, new Comparator<Map.Entry<String, Double>>() {
			public int compare(Map.Entry<String, Double> o1,
					Map.Entry<String, Double> o2) {
				return o2.getValue().compareTo(o1.getValue());
			}
		});
		// Convert sorted map back to a Map
		HashMap<String, Double> sortedMap = new LinkedHashMap<String, Double>();
		for (Iterator<Map.Entry<String, Double>> it = list.iterator(); it.hasNext();) {
			Map.Entry<String, Double> entry = it.next();
			sortedMap.put(entry.getKey(), entry.getValue());
		}
		return sortedMap;
	}
}