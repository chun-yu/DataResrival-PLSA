import java.io.*;
import java.util.Arrays;

public class Main{

	public void readQueryText(PLSA vp,String filename){	
		File fi = new File(filename);
		String line;
        if( fi.exists()){
			try {
				FileReader fr=new FileReader(filename);
				BufferedReader br=new BufferedReader(fr);
				while ((line=br.readLine()) != null) {
					vp.storeQuery(line);
				}
				br.close();
			}
			catch (IOException e) {System.out.println(e);}
		}
	}
	public void readDocText(PLSA vp,String filename){	
		File fi = new File(filename);
		String line;
        if( fi.exists()){
			try {
				FileReader fr=new FileReader(filename);
				BufferedReader br=new BufferedReader(fr);
				while ((line=br.readLine()) != null) {
					vp.storeDoc(line);
				}
				br.close();
			}
			catch (IOException e) {System.out.println(e);}
		}
	}
	
	public static void main(String args[]){	
	long time1, time2;
	long sec=0;long min=0;
	time1 = System.currentTimeMillis();
		Main main = new Main();
		PLSA vp = new PLSA();
		main.readDocText(vp,"doc_list.txt");
		vp.setDocTF();
		main.readQueryText(vp,"query_list.txt");
		vp.retrieve();
	time2 = System.currentTimeMillis();
	sec = (time2-time1)/1000;
	if(sec>60){
		min=sec/60;
		sec=sec%60;
	}
	System.out.println("Time spend: " + min +" : "+ sec );
	}
	
	
}