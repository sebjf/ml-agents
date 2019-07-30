using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class CrawlerAcademy : Academy
{
    private Terrain terrain;
    private TerrainGenerator terrainGenerator;

    public override void InitializeAcademy()
    {
        Monitor.verticalOffset = 1f;
        Physics.defaultSolverIterations = 12;
        Physics.defaultSolverVelocityIterations = 12;
        Time.fixedDeltaTime = 0.01333f; // (75fps). default is .2 (60fps)
        Time.maximumDeltaTime = .15f; // Default is .33

        terrain = FindObjectOfType<Terrain>();
        terrainGenerator = terrain.GetComponent<TerrainGenerator>();

        SetResetParameters();
    }

    public override void AcademyReset()
    {
        SetResetParameters();
    }

    public override void AcademyStep()
    {
    }

    void SetTerrain()
    {
        terrainGenerator.scale = resetParameters["terrainScale"];
        terrainGenerator.depth = resetParameters["terrainDepth"];
        terrainGenerator.GenerateTerrain(terrain.terrainData);
    }

    void SetGravity()
    {
        Physics.gravity = new Vector3(0, -resetParameters["gravity"], 0);
    }

    public void SetResetParameters()
    {
        SetTerrain();
        SetGravity();
    }
}
