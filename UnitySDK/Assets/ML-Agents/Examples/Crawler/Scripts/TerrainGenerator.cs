using UnityEngine;

public class TerrainGenerator : MonoBehaviour
{
    //depth determines the max height of the noise; should be between 0.07 to 1
    public float depth = 0.5f;
    //scale determines how noisy the terrain is
    public float scale = 20f;

    //these should be fixed as they match the dimensions of the 
    private int width = 1500;
    private int length = 600;

    public bool randomizeTerrain;

    private float offsetX;
    private float offsetY;

    private void Start()
    //private void Update()
    {
        Terrain terrain = GetComponent<Terrain>();
        //terrain.terrainData = GenerateTerrain(terrain.terrainData);
    }

    public TerrainData GenerateTerrain(TerrainData terrainData)
    {
        offsetX = 0f;
        offsetY = 0f;
        if (randomizeTerrain)
        {
            offsetX = Random.Range(0f, 99999f);
            offsetY = Random.Range(0f, 99999f);
        }
        terrainData.heightmapResolution = 2049;

        terrainData.size = new Vector3(width, depth, length);

        terrainData.SetHeights(0, 0, GenerateHeights());

        return terrainData;
    }

    float[,] GenerateHeights()
    {
        float[,] heights = new float[width, length];

        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < length; y++)
            {
                heights[x, y] = Calculatelength(x, y);
            }

        }

        return heights;
    }

    float Calculatelength(int x, int y)
    {
        float xCoord = (float)x / width * scale + offsetX;
        float yCoord = (float)y / length * scale + offsetY;

        return Mathf.PerlinNoise(xCoord, yCoord);
    }
}
