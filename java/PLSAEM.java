import java.util.*;
import java.io.*;
import java.lang.Object;
import java.lang.*;
import java.util.Random;
import java.math.BigDecimal; 
public class PLSAEM{
    private int K = 10; // number of topics
	private static Map<String, String> indexMap = new HashMap<String, String>();
	private static Map<String, Integer> indexIDFMap = new HashMap<String, Integer>(); //index idf to doc
	private static Map<String, String> docMap = new HashMap<String, String>();
	private static Map<String, HashMap<String, Integer>> tfMap = new HashMap<String, HashMap<String, Integer>>();
	private static Map<String, HashMap<String, Integer>> DoctfMap = new HashMap<String, HashMap<String, Integer>>(); //doc word tf
	private static Map<String, Integer> docWordNumMap = new HashMap<String, Integer>();
	//pwt ptd map set
	private static Map<String, HashMap<String, Double>> P_W_Tmap = new HashMap<String, HashMap<String, Double>>(); //pwt map  Tk x Wi
	private static Map<String, HashMap<String, Double>> P_T_Dmap = new HashMap<String, HashMap<String, Double>>(); //ptd map dj x Tk
	//ptwd map set  dj 2265 x wi x Tk
	private static Map<String, HashMap<String, HashMap<String, Double >>> P_T_W_Dmap = new HashMap<String, HashMap<String, HashMap<String, Double>>>(); 
	//BGL
	private static Map<String, Double> BGLMmap = new HashMap<String, Double>(); //BGLM MAP
	//P_Q_D map
	public static Map<String, Double> Pq_dmap = new HashMap<String, Double>();
	
	public void doplsa(Map<String, String> _indexMap,Map<String, Integer> _indexIDFMap,Map<String, String> _docMap,Map<String, HashMap<String, Integer>> _tfMap
					,Map<String, HashMap<String, Integer>> _DoctfMap,Map<String, Integer> _docWordNumMap){
		
		indexMap = _indexMap;
		indexIDFMap = _indexIDFMap;
		docMap = _docMap;
		tfMap = _tfMap;
		DoctfMap = _DoctfMap;
		docWordNumMap = _docWordNumMap;
		System.out.println("do PLSA !!");
		init();
		EM(30);
		System.out.println("END EM  !!");
		readBGLMText();
	}
	//initial pwt and ptd random then all plus in one
    private void init(){
		System.out.println("Initial !!");
		String wordName;
		HashMap<String, Double> freqMap;
		double allRandom = 0;
		double temp = 0;
		//Initial PWT map  Tk x Wi
		for(int j = 0 ; j<K ; j++){
			freqMap= new HashMap<String, Double>();
			allRandom = 0;
			for(Map.Entry<String, Integer> entry : docWordNumMap.entrySet()){
				wordName = entry.getKey().toString();
				temp = Math.random();
				freqMap.put( wordName , temp );
				allRandom += temp;
			}
			//normalization
			for(Map.Entry<String, Integer> entry : docWordNumMap.entrySet()){
				wordName = entry.getKey().toString();
				temp = freqMap.get(wordName) / allRandom;
				freqMap.put( wordName ,  temp );
			}
			P_W_Tmap.put( Integer.toString(j) , freqMap );
		}
		
		//Initial PTD map  dj x Tk
		for(Map.Entry<String, String> entry : docMap.entrySet()){
			freqMap= new HashMap<String, Double>();
			wordName = entry.getKey().toString();
			allRandom = 0;
			for(int j = 0 ; j<K ; j++){
				temp = Math.random();
				freqMap.put( Integer.toString(j) , temp );
				allRandom += temp;
			}
			//normalization
			for(int j = 0 ; j<K ; j++){
				temp = freqMap.get(Integer.toString(j)) / allRandom;
				freqMap.put( Integer.toString(j) ,  temp );
			}
			P_T_Dmap.put( wordName , freqMap );
		}
	}
    private void EM(int iters){
		System.out.println("EM STEP !!");
        for (int it = 0; it < iters; it++){
         	// E-step
			Estep();
            // M-step
			Mstep();
			System.out.println( (it+1) +" EM  !!");
        }
    }
 
    private void Estep(){
		//System.out.println("E STEP !!");
		//2265 x dj.size x Tk  normalization for all Tk
		double pwt = 0;
		double ptd = 0;
		double allp = 0;
		double temp = 0;
		String[] word;String docName;
		HashMap<String, Double> freqMap = new HashMap<String, Double>();
		HashMap<String, Double> freqMap2 = new HashMap<String, Double>();
		
		HashMap<String, HashMap<String, Double >> frePMap;
		HashMap<String, Double> frePMap2;
		
		for(Map.Entry<String, String> entry : docMap.entrySet()){
			docName = entry.getKey().toString();
			word = entry.getValue().toString().split(" ");
			
			freqMap2 = P_T_Dmap.get(docName);
			
			frePMap = new HashMap<String, HashMap<String, Double >>();
			
			for(String s: word){
				allp = 0;
				frePMap2 = new HashMap<String, Double >();
				for(int j = 0 ; j<K ; j++){
					freqMap = P_W_Tmap.get(Integer.toString(j));
					pwt = freqMap.get(s);
					ptd = freqMap2.get(Integer.toString(j));
					allp = allp + ( pwt * ptd);
					frePMap2.put(Integer.toString(j),pwt*ptd);
				}
				//normalization
				for(int j = 0 ; j<K ; j++){
					if( allp <= 0)		allp = 1e-6;
					temp = frePMap2.get(Integer.toString(j)) / allp;
					frePMap2.put( Integer.toString(j) ,  temp );
				}
				frePMap.put(s,frePMap2);
			}
			P_T_W_Dmap.put( docName , frePMap );
		}
    }
 
    private void Mstep()
    {
		//System.out.println("M STEP !!");
		//======================================================
		//P_W_Tk  Tk x Wi  normalization for all word to all doc
		double cWD = 0;
		double ptwd = 0;
		double allUPp = 0;
		double allDOWNp = 0;
		double temp = 0;
		String word;String docName;
		HashMap<String, Double> frePMap;
		for(int j = 0 ; j<K ; j++){
			frePMap= new HashMap<String, Double>();
			allDOWNp = 0;
			for(Map.Entry<String, Integer> entry : docWordNumMap.entrySet()){
				word = entry.getKey().toString();
				allUPp = 0;
				//up 
				for(Map.Entry<String, String> entry2 : docMap.entrySet()){
					docName = entry2.getKey().toString();
					if(DoctfMap.get(docName).containsKey(word)){
						cWD = DoctfMap.get(docName).get(word);
						ptwd = P_T_W_Dmap.get(docName).get(word).get(Integer.toString(j));
						allUPp = allUPp + ( cWD*ptwd );
					}
				}
				allDOWNp = allDOWNp + allUPp;
				frePMap.put(word,allUPp);
			}

			//down normalization
			for(Map.Entry<String, Integer> entry : docWordNumMap.entrySet()){
				word = entry.getKey().toString();
				if( allDOWNp <= 0)		allDOWNp = 1e-6;
				temp = frePMap.get(word) / allDOWNp;
				frePMap.put(word,temp);
			}
			P_W_Tmap.put( Integer.toString(j) , frePMap );
		}
		
		//======================================================
		//P_T_dj  dj x Tk  normalization for all word  in a doc
		double cWD2 = 0;
		double ptwd2 = 0;
		double allUPp2 = 0;
		double allDOWNp2 = 0;
		double temp2 = 0;
		String[] word2;String docName2;
		HashMap<String, Double> frePMap2;
		for(Map.Entry<String, String> entry : docMap.entrySet()){
			docName2 = entry.getKey().toString();
			word2 = entry.getValue().toString().split(" ");
			frePMap2 = new HashMap<String, Double>();
			for(int j = 0 ; j<K ; j++){
				allUPp2 = 0;allDOWNp2 = 0;
				for(String s: word2){
					if(DoctfMap.get(docName2).containsKey(s)){
						cWD2 = DoctfMap.get(docName2).get(s);
						ptwd2 = P_T_W_Dmap.get(docName2).get(s).get(Integer.toString(j));
						allUPp2 = allUPp2 + ( cWD2*ptwd2 );
						allDOWNp2 = allDOWNp2+cWD2;
					}
				}
				    if( allDOWNp2 <= 0)		allDOWNp2 = 1e-6;
				temp2 = allUPp2 / allDOWNp2;
				frePMap2.put(Integer.toString(j),temp2);
			}
			P_T_Dmap.put( docName2 , frePMap2 );
		}
    }
 

	public void readBGLMText(){ //read bgla and store
		System.out.println("ReadBGLMText !!");
		File fi = new File("BGLM.txt");
		String line;
        if( fi.exists()){
			try {
				FileReader fr=new FileReader("BGLM.txt");
				BufferedReader br=new BufferedReader(fr);
				while ((line=br.readLine()) != null) {
					String[] terms = line.split("   ");
					double bg = Math.exp(Double.parseDouble(terms[1]));
					BGLMmap.put(terms[0],bg);
				}
				br.close();
			}
			catch (IOException e) {System.out.println(e);}
		}
	}
	public Map<String, Double> getPlsa5(String queryName){ //plsa5  p(q|dj)
		//System.out.println("PLSA ALL !!");
		HashMap<String, Integer> frePMap = tfMap.get(queryName);
		double a = 0.3;
		double b = 0.5;
		Pq_dmap = new HashMap<String, Double>(); //Pq_d map
			String term = indexMap.get(queryName);
			String[] terms = term.split(" ");
		double multi = 0;double bgl = 0; double weight = 0;
		//BigDecimal one = new BigDecimal("1");
		String word;String doc;
		for (Map.Entry<String, String> entry : docMap.entrySet()){
			doc = entry.getKey().toString();
			//one = new BigDecimal("1");
			weight = 0;
			for (Map.Entry<String, Integer> entry2 : frePMap.entrySet()){
				word = entry2.getKey().toString();
				bgl = 0;
				if(BGLMmap.containsKey(word)){
					bgl = BGLMmap.get(word);
				}
				multi = a*getPw_d(word,doc) + b*getEM(word,doc) + (1-a-b)*bgl;
				
					//BigDecimal b2 = new BigDecimal(Double.toString(multi)); 
					//one = one.multiply(b2);
				weight = weight + Math.log(multi)*entry2.getValue();
			}
			Pq_dmap.put(doc , weight);
		}
		return Pq_dmap;
	}
	public double getPw_d(String word,String docName){
		String term = docMap.get(docName);
		String[] terms = term.split(" ");
		int num = terms.length;
		int count = 0;
		HashMap<String, Integer> freqMap = new HashMap<String, Integer>();
	    if(DoctfMap.containsKey(docName)){
			freqMap = DoctfMap.get(docName);
			if(freqMap.containsKey(word)) count = freqMap.get(word);
		}
		return count/num;
	}
	public double getEM(String word,String docName){ // for all 1~K to p(wi|T)p(T|dj)
		double weight = 0;
		HashMap<String, Double> freqMap;
		HashMap<String, Double> freqMap2;
		freqMap2 = P_T_Dmap.get(docName);
		double pwt = 0;
		double ptd = 0;
		for(int i =0;i<K;i++){
			freqMap = P_W_Tmap.get(Integer.toString(i));
			pwt = 0;
			if(freqMap.containsKey(word)){
				pwt = freqMap.get(word);
			}
				ptd = freqMap2.get(Integer.toString(i));
			weight = weight + ( pwt * ptd);
		}
		return weight;
	}
}
