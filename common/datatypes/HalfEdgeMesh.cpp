//
// Created by Seth Parker on 6/17/16.
//

#include "HalfEdgeMesh.h"

namespace volcart {
    ///// Constructors & Destructors /////
    HalfEdgeMesh::HalfEdgeMesh() { };
    HalfEdgeMesh::~HalfEdgeMesh() { clear(); };

    void HalfEdgeMesh::clear() {
        _verts.clear();
        _edges.clear();
        _faces.clear();
    }

    ///// Mesh Access /////
    // Add a vertex at XYZ
    HalfEdgeMesh::VertPtr HalfEdgeMesh::addVert(double x, double y, double z) {
      HalfEdgeMesh::VertPtr v = std::make_shared<HalfEdgeMesh::Vert>();
      v->id = _verts.size();
      if( _verts.size() > 0 )
        v->nextlink = _verts.back();

      v->xyz = cv::Vec3d(x,y,z);
      v->uv  = cv::Vec2d(0,0);

      _verts.push_back(v);

      return v;
    }

    // Add a face connecting v0 -> v1 -> v2
    HalfEdgeMesh::FacePtr HalfEdgeMesh::addFace(IDType v0, IDType v1, IDType v2) {
      // Make the faces, edges, and angles
      HalfEdgeMesh::FacePtr f = std::make_shared<Face>();
      HalfEdgeMesh::EdgePtr e0 = std::make_shared<Edge>();
      HalfEdgeMesh::EdgePtr e1 = std::make_shared<Edge>();
      HalfEdgeMesh::EdgePtr e2 = std::make_shared<Edge>();
      HalfEdgeMesh::AnglePtr a0 = std::make_shared<Angle>();
      HalfEdgeMesh::AnglePtr a1 = std::make_shared<Angle>();
      HalfEdgeMesh::AnglePtr a2 = std::make_shared<Angle>();

      // Link the edges to the face
      f->edge = e0;
      e0->face = e1->face = e2->face = f;

      // Link the edges to each other
      e0->next = e1;
      e1->next = e2;
      e2->next = e0;

      // Link the edges and angles
      e0->angle = a0; a0->edge = e0;
      e1->angle = a1; a1->edge = e1;
      e2->angle = a2; a2->edge = e2;

      // Link the edges to their vertices
      e0->vert = _verts[v0];
      e1->vert = _verts[v1];
      e2->vert = _verts[v2];

      // Link the vertices to their edges if they dont have one
      if ( e0->vert->edge == nullptr ) e0->vert->edge = e0;
      if ( e1->vert->edge == nullptr ) e1->vert->edge = e1;
      if ( e2->vert->edge == nullptr ) e2->vert->edge = e2;

      // Compute the current angles
      cv::Vec3d a;
      a[0] = _angle( e0->vert->xyz, e1->vert->xyz, e2->vert->xyz );
      a[1] = _angle( e1->vert->xyz, e2->vert->xyz, e0->vert->xyz );
      a[2] = _angle( e2->vert->xyz, e0->vert->xyz, e1->vert->xyz );

      // Clamping
      for( int i = 0; i < 3; ++i ) {
        if ( a[i] < MINANGLE )
          a[i] = MINANGLE;
        else if ( a[i] > MAXANGLE )
          a[i] = MAXANGLE;
      }

      // Assign the most recent values
      a0->alpha = a0->beta = a[0];
      a1->alpha = a1->beta = a[1];
      a2->alpha = a2->beta = a[2];

      // Add everything to their respective arrays
      e0->id = _edges.size();
      if(_edges.size() > 0)
      e0->nextlink = _edges.back();
      _edges.push_back(e0);

      e1->id = _edges.size();
      e1->nextlink = _edges.back();
      _edges.push_back(e1);

      e2->id = _edges.size();
      e2->nextlink = _edges.back();
      _edges.push_back(e2);

      f->id = _faces.size();
      if(_faces.size() > 0)
        f->nextlink = _faces.back();
      f->connected = false;
      _faces.push_back(f);

      // Always connect the front face to the last face
      _faces.front()->nextlink = _faces.back();

      return f;
    }

    // Get a vertex by vertex id
    HalfEdgeMesh::VertPtr HalfEdgeMesh::getVert(IDType id) { return _verts[id]; };
    // Get an edge by edge id
    HalfEdgeMesh::EdgePtr HalfEdgeMesh::getEdge(IDType id) { return _edges[id]; };
    // Get a face by face id
    HalfEdgeMesh::FacePtr HalfEdgeMesh::getFace(IDType id) { return _faces[id]; };

    ///// Special Construction Tasks /////
    void HalfEdgeMesh::constructConnectedness() {
      // Connect pairs
      _connectAllPairs();

      // Connect boundaries
      _computeBoundary();
    }

    // Connect the edges that have the same vertices as end points
    void HalfEdgeMesh::_connectAllPairs() {
      HalfEdgeMesh::EdgePtr e0, e1, e2;

      for( auto face = _faces.begin(); face != _faces.end(); ++face ) {
        if( (*face)->connected ) continue;

        e0 = (*face)->edge;
        e1 = e0->next;
        e2 = e1->next;

        e0->pair = _findEdgePair( e0->vert->id, e1->vert->id );
        if ( e0->pair != nullptr ) e0->pair->pair = e0;
        e1->pair = _findEdgePair( e1->vert->id, e2->vert->id );
        if ( e1->pair != nullptr ) e1->pair->pair = e1;
        e2->pair = _findEdgePair( e2->vert->id, e0->vert->id );
        if ( e2->pair != nullptr ) e2->pair->pair = e2;

        (*face)->connected = true;
      }
    }

    // Find the other edge that shares the same two vertices
    HalfEdgeMesh::EdgePtr HalfEdgeMesh::_findEdgePair( HalfEdgeMesh::IDType A, HalfEdgeMesh::IDType B ) {
      HalfEdgeMesh::EdgePtr pair = nullptr;
      // Check these id's against each edge
      for( auto it = _edges.begin(); it != _edges.end(); ++it ) {
        // If the current edge's first if matches this one's B, and vice versa, it's the pair we want
        if( ((*it)->vert->id == B ) && ((*it)->next->vert->id == A ) ) {
          pair = (*it);
          break;
        }
      }

      return pair;
    }

    // Compute which edges are boundaries and which are interior
    void HalfEdgeMesh::_computeBoundary() {
      for( auto it = _verts.begin(); it != _verts.end(); ++it ) {
        if ((*it)->interior())
          _interior.push_back(*it);
        else
          _boundary.push_back(*it);
      }
    }

    ///// Topology Traversal /////
    HalfEdgeMesh::EdgePtr HalfEdgeMesh::nextWheelEdge(EdgePtr e) { return e->next->next->pair; };

    HalfEdgeMesh::EdgePtr HalfEdgeMesh::prevWheelEdge(EdgePtr e) { return (e->pair) ? e->pair->next : nullptr; };

    HalfEdgeMesh::EdgePtr HalfEdgeMesh::nextBoundaryEdge(EdgePtr e) { return e->next->vert->edge; };

    HalfEdgeMesh::EdgePtr HalfEdgeMesh::prevBoundaryEdge(EdgePtr e) {
      HalfEdgeMesh::EdgePtr we = e, last;

      do {
        last = we;
        we = nextWheelEdge(we);
      } while (we && (we != e));

      return last->next->next;
    }


    ///// Math Functions /////
    // Returns the angle between ab and ac
    double HalfEdgeMesh::_angle(cv::Vec3d A, cv::Vec3d B, cv::Vec3d C) {

      cv::Vec3d vec_1 = B - A;
      cv::Vec3d vec_2 = C - A;

      cv::normalize(vec_1, vec_1);
      cv::normalize(vec_2, vec_2);

      double dot = vec_1.dot(vec_2);

      if ( dot <= -1.0f )
        return M_PI;
      else if ( dot >= 1.0f )
        return 0.0f;
      else
        return acos(dot);
    }
}